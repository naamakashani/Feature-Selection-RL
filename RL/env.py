import numpy as np
import os
from sklearn.model_selection import train_test_split
import gym
import torch
import RL.utils as utils
from RL.guesser import Guesser


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


class myEnv(gym.Env):

    def __init__(self,
                 flags,
                 device,
                 oversample=True,
                 load_pretrained_guesser=True):
        self.guesser = Guesser()
        episode_length = flags.episode_length
        self.device = device
        X, y = balance_class(self.guesser.X, self.guesser.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.guesser.X, self.guesser.y,
                                                                                test_size=0.3)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train,
                                                                              self.y_train,
                                                                              test_size=0.05)
        # self.class_0_total = [index for index, value in enumerate(self.y_train) if value == 0]
        # self.class_1_total = [index for index, value in enumerate(self.y_train) if value == 1]

        # val_test_number = int(min(len(self.class_0_total), len(self.class_1_total))*0.2)
        #
        # self.class_0_train = self.class_0_total[:-val_test_number]
        # self.class_1_train = self.class_1_total[:-val_test_number]
        # self.class_0_val = self.class_0_total[-val_test_number:-int(val_test_number/2)]
        # self.class_1_val = self.class_1_total[-val_test_number:-int(val_test_number/2)]
        # self.class_0_test = self.class_0_total[-int(val_test_number/2):]
        # self.class_1_test = self.class_1_total[-int(val_test_number/2):]

        self.episode_length = episode_length
        self.action_probs = utils.diabetes_prob_actions()
        # Load pre-trained guesser network, if needed
        if load_pretrained_guesser:
            save_dir = os.path.join(os.getcwd(), 'model_guesser')
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
        self.state = np.concatenate([np.zeros(self.guesser.features_size), np.zeros(self.guesser.features_size)])

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

    # def reset(self,
    #           mode='training',
    #           patient=0,
    #           train_guesser=True):
    #     """
    #
    #     Args: mode: training / val / test
    #           patient (int): index of patient
    #           train_guesser (Boolean): flag indicating whether to train guesser network in this episode
    #
    #     Selects a patient (random for training, or pre-defined for val and test) ,
    #     Resets the state to contain the basic information,
    #     Resets 'done' flag to false,
    #     Resets 'train_guesser' flag
    #     """
    #
    #
    #     self.state = np.concatenate([np.zeros(self.guesser.features_size), np.zeros(self.guesser.features_size)])
    #
    #     if mode == 'training':
    #         if np.random.rand() < 0.5:
    #             ind = np.random.randint(len(self.class_0_train))
    #             self.patient = self.class_0_train[ind]
    #             self.current_class = 0
    #         else:
    #             ind = np.random.randint(len(self.class_1_train))
    #             self.patient = self.class_1_train[ind]
    #             self.current_class = 1
    #     if mode == 'val':
    #         if np.random.rand() < 0.5:
    #             ind = np.random.randint(len(self.class_0_val))
    #             self.patient = self.class_0_val[ind]
    #             self.current_class = 0
    #         else:
    #             ind = np.random.randint(len(self.class_1_train))
    #             self.patient = self.class_1_train[ind]
    #             self.current_class = 1
    #     if mode == 'test':
    #         if np.random.rand() < 0.5:
    #             ind = np.random.randint(len(self.class_0_test))
    #             self.patient = self.class_0_test[ind]
    #             self.current_class = 0
    #         else:
    #             ind = np.random.randint(len(self.class_1_test))
    #             self.patient = self.class_1_test[ind]
    #             self.current_class = 1
    #
    #     self.done = False
    #     self.s = np.array(self.state)
    #     self.time = 0
    #     if mode == 'training':
    #         self.train_guesser = train_guesser
    #     else:
    #         self.train_guesser = False
    #     return self.s

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
        if self.time == self.episode_length:
            self.terminate_episode()

        return self.s, self.reward, self.done, self.guess

    # Update 'done' flag when episode terminates
    def terminate_episode(self):
        self.done = True

    def update_state(self, action, mode, mask):
        next_state = np.array(self.state)

        if action < self.guesser.features_size:  # Not making a guess
            if mode == 'training':
                next_state[action] = self.X_train[self.patient, action]
            elif mode == 'val':
                next_state[action] = self.X_val[self.patient, action]
            elif mode == 'test':
                next_state[action] = self.X_test[self.patient, action]
            next_state[action + self.guesser.features_size] += 1.
            self.guess = -1
            self.done = False

        else:  # Making a guess
            guesser_input = torch.Tensor(
                self.state[:self.guesser.features_size])
            if torch.cuda.is_available():
                guesser_input = guesser_input.cuda()
            self.guesser.train(mode=False)
            self.probs = self.guesser(guesser_input)
            self.guess = torch.argmax(self.probs).item()
            self.correct_prob = self.probs[self.y_train[self.patient]].item()
            self.terminate_episode()

        return next_state

    def compute_reward(self, mode):
        """ Compute the reward """

        if mode == 'test':
            return None

        if self.guess == -1:  # no guess was made
            return .01 * np.random.rand()
        else:
            reward = self.correct_prob

        if mode == 'training':
            y_true = self.y_train[self.patient]

        if self.train_guesser:
            # train guesser
            self.guesser.optimizer.zero_grad()
            y = torch.Tensor([y_true]).long()
            y = y.to(device=self.device)
            self.guesser.train(mode=True)
            self.guesser.loss = self.guesser.criterion(self.probs, y)
            self.guesser.loss.backward()
            self.guesser.optimizer.step()
            # update learning rate
            self.guesser.update_learning_rate()

        return reward
