import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--hidden-dim1",
                    type=int,
                    default=64,
                    help="Hidden dimension")
parser.add_argument("--hidden-dim2",
                    type=int,
                    default=128,
                    help="Hidden dimension")
FLAGS = parser.parse_args(args=[])


class Guesser(nn.Module):
    """
    implements a net that guesses the outcome given the state
    """

    def __init__(self, features_size,
                 hidden_dim1=FLAGS.hidden_dim1, hidden_dim2=FLAGS.hidden_dim2,
                 num_classes=2):

        super(Guesser, self).__init__()
        self.features_size = features_size

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(features_size, hidden_dim1),
            torch.nn.PReLU(),
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim1, hidden_dim2),
            torch.nn.PReLU(),
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim2, hidden_dim2),
            torch.nn.PReLU(),
        )

        # output layer
        self.logits = nn.Linear(hidden_dim2, num_classes)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        logits = self.logits(x)
        if logits.dim() == 2:
            probs = F.softmax(logits, dim=1)
        else:
            probs = F.softmax(logits, dim=-1)
        return probs

    def _to_variable(self, x: np.ndarray) -> torch.Tensor:
        """torch.Variable syntax helper
        Args:
            x (np.ndarray): 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: torch variable
        """
        return torch.autograd.Variable(torch.Tensor(x))



