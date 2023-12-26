import numpy as np
from torch.utils import data
from torchvision import datasets, transforms
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class NeuralNetwork(nn.Module):
    def __init__(self):
        '''
        Declare layers for the model
        '''
        super().__init__()
        self.image_size = 784
        self.fc0 = nn.Linear(self.image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          weight_decay=0.,
                                          lr=1e-4)

    def forward(self, x):
        ''' Forward pass through the network, returns log_softmax values '''

        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)


def mask(images: np.array) -> np.array:
    """ A function that masks the input vector"""
    for j in range(int(len(images))):

        for i in range(784):
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
    train_losses, val_losses = [], []
    for e in range(nepochs):
        running_loss = 0
        running_val_loss = 0
        for images, labels in train_loader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
            # call mask function on images
            images = mask(images)
            # Training pass
            model.train()  # set model in train mode
            model.optimizer.zero_grad()
            output = model(images)
            loss = model.criterion(output, labels)
            loss.backward()
            model.optimizer.step()
            running_loss += loss.item()
            val_loss = 0
            # 6.2 Evalaute model on validation at the end of each epoch.
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.view(images.shape[0], -1)
                # call mask function on images
                images = mask(images)
                # Training pass
                model.train()
                model.eval()
                output = model(images)
                val_loss = model.criterion(output, labels)
                pred = output.max(1, keepdim=True)[1]
                running_val_loss += val_loss.item()

        # 7. track train loss and validation loss
        train_losses.append(running_loss / len(train_loader))
        val_losses.append(running_val_loss / len(val_loader))

        print("Epoch: {}/{}.. ".format(e + 1, nepochs),
              "Training Loss: {:.3f}.. ".format(running_loss / len(train_loader)),
              "Validation Loss: {:.3f}.. ".format(running_val_loss / len(val_loader)))


def test(model, test_loader, mnist_test):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
            images = mask(images)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print('Accuracy of the network on the {} test images: {:.2%}'.format(len(mnist_test), accuracy))


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
    data_path = 'C:\\Users\\kashann\\PycharmProjects\\adaptivFS\\MNIST'
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    # Download and load the data
    mnist_data = datasets.MNIST(data_path, download=True, train=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, download=True, train=False, transform=transform)
    # load all the test data NOT IN BATCHES
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=True)
    train_size = int(len(mnist_data) * 0.8)
    val_size = len(mnist_data) - train_size
    train_set, val_set = data.random_split(mnist_data, [train_size, val_size])

    # 2.1. create data loader for the trainset (batch_size=64, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    # 2.2. create data loader for the valset (batch_size=64, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False)
    model = NeuralNetwork()
    nepochs = 15
    train_model(model, nepochs,
                train_loader, val_loader)

    test(model, test_loader, mnist_test)
    save_model(model, '/model_guesser_mnist')


if __name__ == "__main__":
    main()
