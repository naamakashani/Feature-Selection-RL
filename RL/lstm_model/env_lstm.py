from RL.lstm_model.state import *
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from RL.utils import *
import gymnasium


class myEnv(gymnasium.Env):

    def __init__(self,
                 flags,
                 device):

        self.device = device
        self.embedding_dim = 10
        self.X, self.y, self.question_names, self.features_size =load_ehr()
        self.X, self.y = balance_class(self.X, self.y)
        self.guesser = Guesser(self.embedding_dim * 2)
        self.question_embedding = nn.Embedding(num_embeddings=self.features_size, embedding_dim=self.embedding_dim)
        self.state = State(self.features_size, self.embedding_dim, self.device)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=0.2)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train,
                                                                              self.y_train,
                                                                              test_size=0.1)
        cost_list = np.array(np.ones(self.features_size + 1))
        self.action_probs = torch.from_numpy(np.array(cost_list))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_guesser = optim.Adam(self.guesser.parameters(), lr=flags.lr)
        self.optimizer_state = optim.Adam(self.state.parameters(), lr=flags.lr)
        self.optimizer_embedding = optim.Adam(self.question_embedding.parameters(), lr=flags.lr)
        self.episode_length = 7
        self.count = 0

    def reset(self,
              mode='training',
              patient=0,
              train_guesser=True):

        # Reset state
        self.state.reset_states()
        self.s = self.state.lstm_h.data.cpu().numpy()

        if mode == 'training':
            self.patient = np.random.randint(self.X_train.shape[0])
        else:
            self.patient = patient

        self.done = False
        self.s = np.array(self.s)
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
        mask = torch.ones(self.features_size + 1)
        mask = mask.to(device=self.device)
        return mask

    def step(self,
             action, mask
             , i, mode='training'):
        """ State update mechanism """

        # update state
        next_state = self.update_state(action, mode, mask, i)
        self.s = next_state

        # compute reward
        self.reward = self.compute_reward(mode)

        self.time += 1
        if self.time == self.features_size:
            self.terminate_episode()

        return self.s, self.reward, self.done, self.guess

    # Update 'done' flag when episode terminates
    def terminate_episode(self):
        self.done = True

    def prob_guesser(self, state):
        guesser_input = torch.Tensor(
            state[:self.features_size])
        if torch.cuda.is_available():
            guesser_input = guesser_input.cuda()
        self.guesser.train(mode=False)
        self.probs = self.guesser(guesser_input)
        self.probs = F.softmax(self.probs, dim=1)
        self.guess = torch.argmax(self.probs).item()
        class_index = int(self.y_train[self.patient].item())
        self.correct_prob = self.probs[0, class_index].item()
        return self.correct_prob

    def print_parametrs(self):
        # Print parameters of guesser after optimization step
        print("\nGuesser parameters after optimization step:")
        for param_name, param in self.guesser.named_parameters():
            print(param_name, param)

        # Print parameters of state after optimization step
        print("\nState parameters after optimization step:")
        for param_name, param in self.state.named_parameters():
            print(param_name, param)

        # Print parameters of question_embedding after optimization step
        print("\nQuestion Embedding parameters after optimization step:")
        for param_name, param in self.question_embedding.named_parameters():
            print(param_name, param)

    def update_state(self, action, mode, mask, eps):
        prev_state = self.s

        if action < self.features_size:  # Not making a guess
            if mode == 'training':
                answer = self.X_train[self.patient, action]
            elif mode == 'val':
                answer = self.X_val[self.patient, action]
            elif mode == 'test':
                answer = self.X_test[self.patient, action]
            ind = torch.LongTensor([action]).to(device=self.device)
            question_embedding = self.question_embedding(ind)
            question_embedding = question_embedding.to(device=self.device)
            next_state = self.state(question_embedding, answer)
            next_state = torch.autograd.Variable(torch.Tensor(next_state))
            next_state = next_state.float()
            probs = self.guesser(next_state)
            y_true = self.y_train[self.patient]
            y_tensor = torch.tensor([int(y_true)])
            y_true_tensor = F.one_hot(y_tensor, num_classes=2)
            self.probs = probs.float()
            self.probs = F.softmax(self.probs, dim=1)
            y_true_tensor = y_true_tensor.float()
            self.optimizer_state.zero_grad()
            self.optimizer_guesser.zero_grad()
            self.optimizer_embedding.zero_grad()
            self.loss = self.criterion(self.probs, y_true_tensor)
            self.loss.backward()
            if eps >= 0:
                self.count = self.count + 1
                self.optimizer_state.step()
                self.optimizer_guesser.step()
                self.optimizer_embedding.step()
            self.reward = self.prob_guesser(next_state) - self.prob_guesser(prev_state)
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
