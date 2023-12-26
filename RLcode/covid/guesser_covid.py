import numpy as np
from fastai.data.load import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class NeuralNetwork(nn.Module):
    def __init__(self):
        '''
        Declare layers for the model
        '''
        super().__init__()
        self.features_size = 14
        self.fc0 = nn.Linear(self.features_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          weight_decay=0.,
                                          lr=1e-4)

    def forward(self, x):
        ''' Forward pass through the network, returns log_softmax values '''
        # check if x is numpy array
        if not isinstance(x, np.ndarray):
            x = x.to(self.fc0.weight.dtype)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x)


def mask(images: np.array) -> np.array:
    """ A function that masks the input vector"""
    # check if images has 1 dim
    if len(images.shape) == 1:
        for i in range(14):
            # choose to mask in probability of 0.3
            if (np.random.rand() < 0.3):
                images[i] = 0
        return images
    else:

        for j in range(int(len(images))):
            for i in range(14):
                # choose to mask in probability of 0.3
                if (np.random.rand() < 0.3):
                    images[j][i] = 0
        return images


def train_model(model,
                nepochs, train_loader, val_loader):
    '''
    Train a pytorch model and evaluate it every epoch.
    Params:
    model - a pytorch model to train
    optimizer - an optimizer
    criterion - the criterion (loss function)
    nepochs - number of training epochs
    train_loader - dataloader for the trainset
    val_loader - dataloader for the valset
    is_image_input (default False) - If false, flatten 2d images into a 1d array.
                                  Should be True for Neural Networks
                                  but False for Convolutional Neural Networks.
    '''
    for e in range(nepochs):
        for images, labels in train_loader:
            images = images.view(images.shape[0], -1)
            images = mask(images)
            model.train()  # set model in train mode
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
                    images = mask(images)
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
    model.eval()
    total = 0
    correct = 0
    y_hat = []
    with torch.no_grad():
        for x in X_test:
            x = mask(x)
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
    X, y, names = utils.load_covid()
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
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
    # Define batch size
    batch_size = 32
    # Create a DataLoader
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    # Convert data to PyTorch tensors
    X_tensor_val = torch.Tensor(X_val)
    y_tensor_val = torch.Tensor(y_val)  # Assuming y_data contains integers
    dataset_val = TensorDataset(X_tensor_val, y_tensor_val)
    data_loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    model = NeuralNetwork()
    nepochs = 20
    train_model(model, nepochs,
                data_loader_train, data_loader_val)

    test(model, X_test, y_test)
    save_model(model, 'C:\\Users\\kashann\\PycharmProjects\\adaptivFS\\covid\\model_guesser_covid')


if __name__ == "__main__":
    main()
