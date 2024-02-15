import argparse
import torch
import torch.nn
from torch.optim import lr_scheduler
from dqn import DQN
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--decay_step_size",
                    type=int,
                    default=50000,
                    help="LR decay step size")
parser.add_argument("--lr_decay_factor",
                    type=float,
                    default=0.1,
                    help="LR decay factor")
parser.add_argument("--min_lr",
                    type=float,
                    default=1e-5,
                    help="Minimal learning rate")
FLAGS = parser.parse_args(args=[])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def lambda_rule(i_episode) -> float:
    """ stepwise learning rate calculator """
    exponent = int(np.floor((i_episode + 1) / FLAGS.decay_step_size))
    return np.power(FLAGS.lr_decay_factor, exponent)


class Agent(object):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int, lr, weight_decay) -> None:
        """Agent class that choose action and train
        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            hidden_dim (int): hidden dimension
        """
        self.dqn = DQN(input_dim, output_dim, hidden_dim)
        self.target_dqn = DQN(input_dim, output_dim, hidden_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.optim = torch.optim.Adam(self.dqn.parameters(),
                                      lr=lr,
                                      weight_decay=weight_decay)
        self.scheduler = lr_scheduler.LambdaLR(self.optim,
                                               lr_lambda=lambda_rule)

        self.update_target_dqn()

    def update_target_dqn(self):
        # hard copy model parameters to target model parameters
        for param, target_param in zip(self.dqn.parameters(), self.target_dqn.parameters()):
            target_param.data.copy_(param.data)

    def _to_variable(self, x: np.ndarray) -> torch.Tensor:
        """torch.Variable syntax helper
        Args:
            x (np.ndarray): 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: torch variable
        """
        return torch.autograd.Variable(torch.Tensor(x))

    def get_action(self, states: np.ndarray, env,
                   eps: float,
                   mask: np.ndarray, mode) -> int:
        """Returns an action
        Args:
            states (np.ndarray): 2-D tensor of shape (n, input_dim)
            eps (float): 𝜺-grget_actioneedy for exploration
            mask (np.ndarray) zeroes out q values for questions that were already asked, so they will not be chosen again
        Returns:
            int: action index
        """
        if np.random.rand() < eps and mode == 'training':
            array_probs = env.action_probs * mask
            # make the tensor numpy array
            array_probs = array_probs.cpu().detach().numpy()
            return np.random.choice(self.output_dim, p=array_probs / array_probs.sum())

        else:
            self.dqn.train(mode=False)
            scores = self.get_Q(states)
            _, argmax = torch.max(scores.data * mask, 1)
            return int(argmax.item())

    def get_action_not_guess(self, states: np.ndarray, env,
                             eps: float,
                             mask: np.ndarray, mode) -> int:

        if np.random.rand() < eps and mode == 'training':
            array_probs = env.action_probs[:-1] * mask[:-1]
            # make the tensor numpy array
            array_probs = array_probs.cpu().detach().numpy()
            return np.random.choice(self.output_dim - 1, p=array_probs / array_probs.sum())

        else:
            self.dqn.train(mode=False)
            scores = self.get_Q(states)
            _, argmax = torch.max(scores.data * mask, 1)

            if argmax.item() == self.output_dim-1:
                # choose the second highest value
                scores.data[0][argmax.item()] = 0
                _, argmax = torch.max(scores.data * mask, 1)
            return int(argmax.item())

    def get_Q(self, states: np.ndarray) -> torch.FloatTensor:
        """Returns `Q-value`
        Args:
            states (np.ndarray): 2-D Tensor of shape (n, input_dim)
        Returns:
            torch.FloatTensor: 2-D Tensor of shape (n, output_dim)
        """
        states = self._to_variable(states.reshape(-1, self.input_dim))
        states = states.to(device=device)
        self.dqn.train(mode=False)
        return self.dqn(states)

    def get_target_Q(self, states: np.ndarray) -> torch.FloatTensor:
        """Returns `Q-value`
        Args:
            states (np.ndarray): 2-D Tensor of shape (n, input_dim)
        Returns:
            torch.FloatTensor: 2-D Tensor of shape (n, output_dim)
        """
        states = self._to_variable(states.reshape(-1, self.input_dim))
        self.target_dqn.train(mode=False)
        return self.target_dqn(states)

    def train(self, Q_pred: torch.FloatTensor, Q_true: torch.FloatTensor) -> float:
        """Computes `loss` and backpropagation
        Args:
            Q_pred (torch.FloatTensor): Predicted value by the network,
                2-D Tensor of shape(n, output_dim)
            Q_true (torch.FloatTensor): Target value obtained from the game,
                2-D Tensor of shape(n, output_dim)
        Returns:
            float: loss value
        """
        self.dqn.train(mode=True)
        self.optim.zero_grad()
        loss = self.loss_fn(Q_pred, Q_true)
        loss.backward()
        self.optim.step()
        return loss

    def update_learning_rate(self):
        """ Learning rate updater """
        self.scheduler.step()
        lr = self.optim.param_groups[0]['lr']
        if lr < FLAGS.min_lr:
            self.optim.param_groups[0]['lr'] = FLAGS.min_lr
