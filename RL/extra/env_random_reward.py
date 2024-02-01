import numpy as np
import os
from sklearn.model_selection import train_test_split
import gymnasium
import torch
import RL.utils as utils
from RL.guesser import Guesser
import torch.nn.functional as F


def balance_class(X, y):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    minority_class = unique_classes[np.argmin(class_counts)]
    majority_class = unique_classes[np.argmax(class_counts)]

    # Get indices of samples belonging to each class
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y == majority_class)[0]

    # Calculate the difference in sample counts
    minority_count = len(minority_indices)
    majority_count = len(majority_indices)
    count_diff = majority_count - minority_count

    # Duplicate samples from the minority class to balance the dataset
    if count_diff > 0:
        # Randomly sample indices from the minority class to duplicate
        duplicated_indices = np.random.choice(minority_indices, count_diff, replace=True)
        # Concatenate the duplicated samples to the original arrays
        X_balanced = np.concatenate([X, X[duplicated_indices]], axis=0)
        y_balanced = np.concatenate([y, y[duplicated_indices]], axis=0)
    else:
        X_balanced = X.copy()  # No need for balancing, as classes are already balanced
        y_balanced = y.copy()
    return X_balanced, y_balanced


class myEnv(gymnasium.Env):

    def __init__(self,
                 flags,
                 device,
                 oversample=True,
                 load_pretrained_guesser=True):
        self.guesser = Guesser()
        self.device = device
        # X, y = balance_class(self.guesser.X, self.guesser.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.guesser.X, self.guesser.y,
                                                                                test_size=0.3)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train,
                                                                              self.y_train,
                                                                              test_size=0.05)

        cost_list = np.array(np.ones(self.guesser.features_size + 1))
        self.action_probs = torch.from_numpy(np.array(cost_list))
        # Load pre-trained guesser network, if needed
        if load_pretrained_guesser:
            save_dir = os.path.join(os.getcwd(), '../model_guesser')
            guesser_filename = 'best_guesser.pth'
            guesser_load_path = os.path.join(save_dir, guesser_filename)
            if os.path.exists(guesser_load_path):
                print('Loading pre-trained guesser')
                guesser_state_dict = torch.load(guesser_load_path)
                self.guesser.load_state_dict(guesser_state_dict)

    def reset(self,
              mode='training',
              patient=0,
              train_guesser=True):
        self.state = np.concatenate([np.zeros(self.guesser.features_size)])

        if mode == 'training':
            self.patient = np.random.randint(self.X_train.shape[0])
        else:
            self.patient = patient

        self.done = False
        self.s = np.array(self.state)
        self.time = 0
        if mode == 'training':
            self.train_guesser = train_guesser
        else:
            self.train_guesser = False
        return self.s

    def reset_mask(self):
        """ A method that resets the mask that is applied
        to the q values, so that questions that were already
        asked will not be asked again.
        """
        mask = torch.ones(self.guesser.features_size + 1)
        mask = mask.to(device=self.device)

        return mask

    def step(self,
             action, mask,
             mode='training'):
        """ State update mechanism """

        # update state
        next_state = self.update_state(action, mode, mask)
        self.state = np.array(next_state)
        self.s = np.array(self.state)

        # compute reward
        self.reward = self.compute_reward(mode)

        self.time += 1
        if self.time == self.guesser.features_size:
            self.terminate_episode()

        return self.s, self.reward, self.done, self.guess

    # Update 'done' flag when episode terminates
    def terminate_episode(self):

        self.done = True

    def prob_guesser(self, state):
        guesser_input = torch.Tensor(
            state[:self.guesser.features_size])
        if torch.cuda.is_available():
            guesser_input = guesser_input.cuda()
        self.guesser.train(mode=False)
        self.probs = self.guesser(guesser_input)
        self.guess = torch.argmax(self.probs).item()
        self.correct_prob = self.probs[self.y_train[self.patient]].item()
        return self.correct_prob

    def update_state(self, action, mode, mask):
        prev_state = np.array(self.state)
        next_state = np.array(self.state)
        if action < self.guesser.features_size:  # Not making a guess
            if mode == 'training':
                next_state[action] = self.X_train[self.patient, action]
            elif mode == 'val':
                next_state[action] = self.X_val[self.patient, action]
            elif mode == 'test':
                next_state[action] = self.X_test[self.patient, action]
            # self.reward = self.prob_guesser(next_state) - self.prob_guesser(prev_state)
            self.reward = .01 * np.random.rand()
            self.guess = -1
            self.done = False
            return next_state

        else:
            self.reward = self.prob_guesser(prev_state)
            self.terminate_episode()
            return prev_state

    def compute_reward(self, mode):
        """ Compute the reward """

        if mode == 'test':
            return None

        if self.guess == -1:  # no guess was made
            return self.reward

        if mode == 'training':
            y_true = self.y_train[self.patient]
            if self.train_guesser:
                self.guesser.optimizer.zero_grad()
                self.guesser.train(mode=True)
                y_tensor = torch.tensor([int(y_true)])
                y_true_tensor = F.one_hot(y_tensor, num_classes=2).squeeze()
                self.probs = self.probs.float()
                y_true_tensor = y_true_tensor.float()
                self.guesser.loss = self.guesser.criterion(self.probs, y_true_tensor)
                self.guesser.loss.backward()
                self.guesser.optimizer.step()

        return self.reward
