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
import RL.utils as utils
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--directory",
                    type=str,
                    default="C:\\Users\\kashann\\PycharmProjects\\choiceMira\\RL",
                    help="Directory for saved models")
parser.add_argument("--batch_size",
                    type=int,
                    default=256,
                    help="Mini-batch size")
parser.add_argument("--num_epochs",
                    type=int,
                    default=10000,
                    help="number of epochs")
parser.add_argument("--hidden-dim1",
                    type=int,
                    default=32,
                    help="Hidden dimension")
parser.add_argument("--hidden-dim2",
                    type=int,
                    default=64,
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
                    default=100,
                    help="Number of validation trials without improvement")

FLAGS = parser.parse_args(args=[])


def add_noise(X, noise_std=0.01):
    """
    Add Gaussian noise to the input features.

    Parameters:
    - X: Input features (numpy array).
    - noise_std: Standard deviation of the Gaussian noise.

    Returns:
    - X_noisy: Input features with added noise.
    """
    noise = np.random.normal(loc=0, scale=noise_std, size=X.shape)
    X_noisy = X + noise
    return X_noisy


def balance_class(X, y, noise_std=0.01):
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

    # Add noise to the features of the minority class to balance the dataset
    if count_diff > 0:
        # Randomly sample indices from the minority class to add noise
        noisy_indices = np.random.choice(minority_indices, count_diff, replace=True)
        # Add noise to the features of the selected samples
        X_balanced = np.concatenate([X, add_noise(X[noisy_indices], noise_std)], axis=0)
        y_balanced = np.concatenate([y, y[noisy_indices]], axis=0)
    else:
        X_balanced = X.copy()  # No need for balancing, as classes are already balanced
        y_balanced = y.copy()
    return X_balanced, y_balanced


def multiply_samples_with_noise(X, Y, noise_std=0.01):
    """
    Multiply samples of input features with noise.

    Parameters:
    - X: Input features (numpy array).
    - Y: Target values (numpy array).
    - num_samples: Number of times to multiply the samples.
    - noise_std: Standard deviation of the Gaussian noise.

    Returns:
    - X_multiplied_noisy: Multiplied input features with added noise.
    - Y: Target values (unchanged).
    """
    X_multiplied_noisy = [X]
    Y_mul = np.concatenate([Y, Y])

    X_noisy = add_noise(X, noise_std)
    X_multiplied_noisy.append(X_noisy)
    X_multiplied_noisy = np.concatenate(X_multiplied_noisy)

    return X_multiplied_noisy, Y_mul


def augment_data(X, Y, num_augmentations=50, noise_level=0.1):
    """
    Augment data by duplicating and adding Gaussian noise.

    Parameters:
    X (numpy.ndarray): Original feature data.
    Y (numpy.ndarray): Original labels.
    num_augmentations (int): Number of times to augment each sample.
    noise_level (float): Standard deviation of Gaussian noise to add.

    Returns:
    X_augmented (numpy.ndarray): Augmented feature data.
    Y_augmented (numpy.ndarray): Augmented labels.
    """
    X_augmented = []
    Y_augmented = []

    for i in range(num_augmentations):
        noise = np.random.normal(0, noise_level, X.shape)
        X_noisy = X + noise
        X_augmented.append(X_noisy)
        Y_augmented.append(Y)

    # Concatenate the original data with the augmented data
    X_augmented = np.vstack([X] + X_augmented)
    Y_augmented = np.hstack([Y] + Y_augmented)

    return X_augmented, Y_augmented


class Guesser(nn.Module):
    """
    implements a net that guesses the outcome given the state
    """

    def __init__(self,
                 hidden_dim1=FLAGS.hidden_dim1, hidden_dim2=FLAGS.hidden_dim2,
                 num_classes=2):

        super(Guesser, self).__init__()
        self.X, self.y, self.question_names, self.features_size = utils.create_demo()
        self.cost= utils.load_cost()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(self.features_size, hidden_dim1),
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
        self.layer4 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim2, hidden_dim2),
            torch.nn.PReLU(),
        )

        self.layer5 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim2, hidden_dim2),
            torch.nn.PReLU(),
        )

        # output layer for regression
        self.output = nn.Linear(hidden_dim2, 1)  # Single output for regression
        self.criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          weight_decay=FLAGS.weight_decay,
                                          lr=FLAGS.lr)
        self.path_to_save = os.path.join(os.getcwd(), 'model_guesser_regression')

    def forward(self, x):
        # replace NaN with 0
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        output = self.output(x)  # Get the regression output
        return output

    def _to_variable(self, x: np.ndarray) -> torch.Tensor:
        """torch.Variable syntax helper
        Args:
            x (np.ndarray): 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: torch variable
        """
        return torch.autograd.Variable(torch.Tensor(x))


def mask(input: np.array) -> np.array:
    '''
    Mask feature of the input
    :param images: input
    :return: masked input
    '''

    # check if images has 1 dim
    if len(input.shape) == 1:
        for i in range(input.shape[0]):
            # choose a random number between 0 and 1
            # fraction = np.random.uniform(0, 1)
            fraction = 0.2
            if (np.random.rand() < fraction):
                input[i] = 0
        return input
    else:
        for j in range(int(len(input))):
            for i in range(input[0].shape[0]):
                # fraction = np.random.uniform(0, 1)
                fraction = 0.2
                if np.random.rand() < fraction:
                    input[j][i] = 0
        return input



def create_adverserial_input(inputs, labels, pretrained_model):
    input = inputs.view(inputs.shape[0], -1).float()
    input.requires_grad_(True)  # Set requires_grad for input
    # Forward pass
    output = pretrained_model(input)
    labels = torch.Tensor(labels).long()
    loss = pretrained_model.criterion(output, labels)

    # Backward pass
    loss.backward()
    #
    # Get the gradients
    gradient = input.grad

    # Identify the most influential features (those with the largest absolute gradients).
    absolute_gradients = torch.abs(gradient)
    max_gradients_indices = torch.argmax(absolute_gradients, dim=-1)

    # Zero out the most influential features.
    mask = torch.nn.functional.one_hot(max_gradients_indices, num_classes=input.shape[-1]).float()
    zeroed_input_features = input * (1 - mask)
    return zeroed_input_features


def val(model, val_loader, best_val_loss=float('inf')):
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for input, labels in val_loader:
            input = mask(input)
            input = input.view(input.shape[0], -1)
            input = input.float()
            output = model(input)
            labels = labels.float().view(-1, 1)  # Reshape labels for regression
            loss = model.criterion(output, labels)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    print(f'Validation Loss: {avg_loss:.4f}')
    if avg_loss <= best_val_loss:
        save_model(model)
    return avg_loss



def test(test_loader, path_to_save):
    guesser_filename = 'best_guesser.pth'
    guesser_load_path = os.path.join(path_to_save, guesser_filename)
    model = Guesser()  # Assuming Guesser is your model class
    guesser_state_dict = torch.load(guesser_load_path)
    model.load_state_dict(guesser_state_dict)
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = mask(images)
            images = images.view(images.shape[0], -1)
            images = images.float()
            output = model(images)

            # Append true and predicted labels for later analysis
            y_true.extend(labels.numpy())  # Assuming labels is a numpy array
            y_pred.extend(output.numpy().flatten())  # Assuming output is a numpy array

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate and print regression metrics
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'Mean Absolute Error: {mae:.4f}')

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
    val_trials_without_improvement = 0
    best_val_auc = 0
    # count total sampels in trainloaser
    accuracy_list = []
    training_loss_list = []
    for i in range(1, nepochs):
        running_loss = 0
        for input, labels in train_loader:
            input = mask(input)
            input = input.view(input.shape[0], -1).float()
            model.train()
            model.optimizer.zero_grad()
            output = model(input)
            labels = labels.float().view(-1, 1)  # Reshape labels for regression
            loss = model.criterion(output, labels)
            running_loss += loss.item()
            loss.backward()
            model.optimizer.step()

        training_loss_list.append(running_loss / len(train_loader))
        # print(f'Epoch: {i}, training loss: {running_loss / len(train_loader):.2f}')
        if i % 20 == 0:
            new_best_val_auc = val(model, val_loader, best_val_auc)
            accuracy_list.append(new_best_val_auc)
            if new_best_val_auc > best_val_auc:
                best_val_auc = new_best_val_auc
                val_trials_without_improvement = 0
            else:
                val_trials_without_improvement += 1
            # check whether to stop training
            if val_trials_without_improvement == FLAGS.val_trials_wo_im:
                print('Did not achieve val AUC improvement for {} trials, training is done.'.format(
                    FLAGS.val_trials_wo_im))
                break
    save_plot_acuuracy_epoch(accuracy_list, training_loss_list)


def save_plot_acuuracy_epoch(val_accuracy_list, training_loss_list):
    epochs_val = range(1, len(val_accuracy_list) + 1)

    plt.figure(figsize=(10, 5))

    # Plot Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_val, val_accuracy_list, marker='o', linestyle='-', color='b')
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    epochs_train = range(1, len(training_loss_list) + 1)
    # Plot Training Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_train, training_loss_list, marker='o', linestyle='-', color='r')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('accuracy_loss_plot.png')
    plt.show()


def save_model(model):
    '''
    Save the model to a given path
    :param model: model to save
    :param path: path to save the model to
    :return: None
    '''
    path = model.path_to_save
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
                                                        test_size=0.2,
                                                        random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.5,
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
    X_tensor_test = torch.Tensor(X_test)
    y_tensor_test = torch.Tensor(y_test)  # Assuming y_data contains integers
    dataset_test = TensorDataset(X_tensor_test, y_tensor_test)
    data_loader_test = DataLoader(dataset_test, batch_size=FLAGS.batch_size, shuffle=True)
    test(data_loader_test, model.path_to_save)


if __name__ == "__main__":
    os.chdir(FLAGS.directory)
    main()
