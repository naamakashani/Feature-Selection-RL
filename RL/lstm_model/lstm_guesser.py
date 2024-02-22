import argparse
import pandas as pd
from itertools import count
from random import random
import numpy as np
from fastai.data.load import DataLoader
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import RL.lstm_model.utils_lstm as utils
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--directory",
                    type=str,
                    default="C:\\Users\\kashann\\PycharmProjects\\choiceMira\\RL\\lstm_model",
                    help="Directory for saved models")
parser.add_argument("--batch_size",
                    type=int,
                    default=16,
                    help="Mini-batch size")
parser.add_argument("--num_epochs",
                    type=int,
                    default=200,
                    help="number of epochs")
parser.add_argument("--hidden-dim1",
                    type=int,
                    default=64,
                    help="Hidden dimension")
parser.add_argument("--hidden-dim2",
                    type=int,
                    default=128,
                    help="Hidden dimension")
parser.add_argument("--lr",
                    type=float,
                    default=1e-4,
                    help="Learning rate")
parser.add_argument("--weight_decay",
                    type=float,
                    default=0.001,
                    help="l_2 weight penalty")
parser.add_argument("--val_trials_wo_im",
                    type=int,
                    default=20,
                    help="Number of validation trials without improvement")

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

    # def train_model(self, input):
    #     '''
    #     Train a pytorch model and evaluate it every 2 epoch.
    #     Params:
    #     model - a pytorch model to train
    #     nepochs - number of training epochs
    #     train_loader - dataloader for the trainset
    #     val_loader - dataloader for the valset
    #     '''
    #     x, y= input[0], input[1]
    #     # x= x.squeeze()
    #     # x = x.view(x.shape[1], -1).float()
    #     self.train()
    #     self.optimizer.zero_grad()
    #     #create y long tensor put 1 in the correct class
    #     x = self._to_variable(x)
    #     x.requires_grad_(True)
    #     output = self.forward(x)
    #     labels = torch.zeros(2)
    #     labels[y] = 1
    #     #transpose the labels
    #     labels = labels.view(1, -1)
    #     loss = self.criterion(output, labels)
    #     loss.backward(retain_graph=True)
    #     self.optimizer.step()

