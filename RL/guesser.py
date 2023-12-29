import argparse

import numpy as np
from fastai.data.load import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import RL.utils as utils

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--directory",
                    type=str,
                    default="C:\\Users\\kashann\\PycharmProjects\\choiceMira\\RL",
                    help="Directory for saved models")

parser.add_argument("--batch_size",
                    type=int,
                    default=64,
                    help="Mini-batch size")
parser.add_argument("--num_epochs",
                    type=int,
                    default=400,
                    help="number of epochs")
parser.add_argument("--hidden-dim",
                    type=int,
                    default=64,
                    help="Hidden dimension")
parser.add_argument("--capacity",
                    type=int,
                    default=10000,
                    help="Replay memory capacity")
parser.add_argument("--max-episode",
                    type=int,
                    default=2000,
                    help="e-Greedy target episode (eps will be the lowest at this episode)")
parser.add_argument("--min-eps",
                    type=float,
                    default=0.01,
                    help="Min epsilon")
parser.add_argument("--lr",
                    type=float,
                    default=1e-4,
                    help="Learning rate")
parser.add_argument("--weight_decay",
                    type=float,
                    default=0e-4,
                    help="l_2 weight penalty")
parser.add_argument("--val_interval",
                    type=int,
                    default=1000,
                    help="Interval for calculating validation reward and saving model")
parser.add_argument("--episode_length",
                    type=int,
                    default=7,
                    help="Episode length")
# Environment params
parser.add_argument("--g_hidden-dim",
                    type=int,
                    default=256,
                    help="Guesser hidden dimension")
parser.add_argument("--g_weight_decay",
                    type=float,
                    default=0e-4,
                    help="Guesser l_2 weight penalty")

FLAGS = parser.parse_args(args=[])


class Guesser(nn.Module):
    def __init__(self):
        '''
        Declare layers for the model
        '''
        super().__init__()
        self.X, self.y, self.question_names, self.features_size = utils.load_data_labels()
        self.fc0 = nn.Linear(self.features_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          weight_decay=0.,
                                          lr=1e-4)

        self.path_to_save = os.path.join(os.getcwd(), 'model_guesser')

    def forward(self, x):
        ''' Forward pass through the network, returns log_softmax values '''
        if not isinstance(x, np.ndarray):
            x = x.to(self.fc0.weight.dtype)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x)


def mask(model, images: np.array) -> np.array:
    '''
    Mask feature of the input
    :param images: input
    :return: masked input
    '''
    # check if images has 1 dim
    if len(images.shape) == 1:
        for i in range(model.features_size):
            # choose to mask in probability of 0.3
            if (np.random.rand() < 0.3):
                images[i] = 0
        return images
    else:

        for j in range(int(len(images))):
            for i in range(model.features_size):
                # choose to mask in probability of 0.3
                if (np.random.rand() < 0.3):
                    images[j][i] = 0
        return images


def train_model(model,
                nepochs, train_loader, val_loader):
    '''
    Train a pytorch model and evaluate it every 2 epoch.
    Params:
    model - a pytorch model to train
    nepochs - number of training epochs
    train_loader - dataloader for the trainset
    val_loader - dataloader for the valset
    '''
    for e in range(nepochs):
        for images, labels in train_loader:
            images = images.view(images.shape[0], -1)
            images = mask(model, images)
            model.train()
            model.optimizer.zero_grad()
            output = model(images)
            y_pred = []
            for y in labels:
                y = torch.Tensor(y).long()
                y_pred.append(y)
            labels = torch.Tensor(np.array(y_pred)).long()
            loss = model.criterion(output, labels)
            loss.backward()
            model.optimizer.step()
        with torch.no_grad():
            if e % 2 == 0:
                for images, labels in val_loader:
                    images = images.view(images.shape[0], -1)
                    # call mask function on images
                    images = mask(model, images)
                    # Training pass
                    model.eval()
                    output = model(images)
                    y_pred = []
                    for y in labels:
                        y = torch.Tensor(y).long()
                        y_pred.append(y)
                    labels = torch.Tensor(np.array(y_pred)).long()

                    val_loss = model.criterion(output, labels)
                    print("val_loss: ", val_loss.item())


def test(model, X_test, y_test):
    '''
    Test the model on the test set
    :param model: model to test
    :param X_test: data to test on
    :param y_test: labels to test on
    :return: accuracy of the model
    '''
    model.eval()
    total = 0
    correct = 0
    y_hat = []
    with torch.no_grad():
        for x in X_test:
            x = mask(model, x)
            # create tensor form x
            x = torch.Tensor(x).float()
            output = model(x)
            _, predicted = torch.max(output.data, 0)
            y_hat.append(predicted)

    # compare y_hat to y_test
    y_hat = [tensor.item() for tensor in y_hat]
    for i in range(len(y_hat)):
        if y_hat[i] == y_test[i]:
            correct += 1
        total += 1
    accuracy = correct / total
    print('Accuracy of the network on the {} test images: {:.2%}'.format(len(X_test), accuracy))


def save_model(model, path):
    '''
    Save the model to a given path
    :param model: model to save
    :param path: path to save the model to
    :return: None
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    guesser_filename = 'best_guesser.pth'
    guesser_save_path = os.path.join(path, guesser_filename)
    # save guesser
    if os.path.exists(guesser_save_path):
        os.remove(guesser_save_path)
    torch.save(model.cpu().state_dict(), guesser_save_path + '~')
    os.rename(guesser_save_path + '~', guesser_save_path)


def main():
    '''
    Train a neural network to guess the correct answer
    :return:
    '''
    model = Guesser()
    X_train, X_test, y_train, y_test = train_test_split(model.X,
                                                        model.y,
                                                        test_size=0.33,
                                                        random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.05,
                                                      random_state=24)
    # Convert data to PyTorch tensors
    X_tensor_train = torch.from_numpy(X_train)

    y_tensor_train = torch.from_numpy(y_train)  # Assuming y_data contains integers
    # Create a TensorDataset
    dataset_train = TensorDataset(X_tensor_train, y_tensor_train)

    # Create a DataLoader
    data_loader_train = DataLoader(dataset_train, batch_size=FLAGS.batch_size, shuffle=True)
    # Convert data to PyTorch tensors
    X_tensor_val = torch.Tensor(X_val)
    y_tensor_val = torch.Tensor(y_val)  # Assuming y_data contains integers
    dataset_val = TensorDataset(X_tensor_val, y_tensor_val)
    data_loader_val = DataLoader(dataset_val, batch_size=FLAGS.batch_size, shuffle=True)
    train_model(model, FLAGS.num_epochs,
                data_loader_train, data_loader_val)

    test(model, X_test, y_test)
    save_model(model, model.path_to_save)


if __name__ == "__main__":
    os.chdir(FLAGS.directory)
    main()
