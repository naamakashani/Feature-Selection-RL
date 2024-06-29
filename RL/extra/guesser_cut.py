import argparse
from itertools import count

import numpy as np
from fastai.data.load import DataLoader
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
                    default=64,
                    help="Mini-batch size")
parser.add_argument("--num_epochs",
                    type=int,
                    default=100,
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
parser.add_argument("--val_interval",
                    type=int,
                    default=50,
                    help="Interval for calculating validation reward and saving model")
parser.add_argument("--val_trials_wo_im",
                    type=int,
                    default=30,
                    help="Number of validation trials without improvement")

FLAGS = parser.parse_args(args=[])



class Guesser(nn.Module):
    """
    implements a net that guesses the outcome given the state
    """

    def __init__(self,
                 hidden_dim1=FLAGS.hidden_dim1, hidden_dim2=FLAGS.hidden_dim2,
                 num_classes=2):

        super(Guesser, self).__init__()
        self.X, self.y, self.question_names, self.features_size = utils.load_data_labels_cut()
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

        # output layer
        self.logits = nn.Linear(hidden_dim2, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          weight_decay=FLAGS.weight_decay,
                                          lr=FLAGS.lr)
        self.path_to_save = os.path.join(os.getcwd(), '../model_guesser')

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


def mask(images: np.array) -> np.array:
    '''
    Mask feature of the input
    :param images: input
    :return: masked input
    '''
    # check if images has 1 dim
    if len(images.shape) == 1:
        for i in range(images.shape[0]):
            # choose to mask in probability of 0.3
            if (np.random.rand() < 0.3):
                images[i] = 0
        return images
    else:

        for j in range(int(len(images))):
            for i in range(images[0].shape[0]):
                # choose to mask in probability of 0.3
                if (np.random.rand() < 0.3):
                    images[j][i] = 0
        return images


def val(model, val_loader, best_val_auc=0):
    correct = 0
    total = 0
    model.eval()
    valid_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.view(images.shape[0], -1)
            # call mask function on images
            images = mask(images)
            images = images.float()
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            y_pred = []
            for y in labels:
                y = torch.Tensor(y).long()
                y_pred.append(y)
            labels = torch.Tensor(np.array(y_pred)).long()
            loss = model.criterion(output, labels)
            valid_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Validation Accuracy: {accuracy:.2f}')
    # if accuracy >= best_val_auc:
    #     save_model(model)
    return accuracy


def test(test_loader, path_to_save):
    guesser_filename = '../model_guesser/best_guesser.pth'
    guesser_load_path = os.path.join(path_to_save, guesser_filename)
    model = Guesser()  # Assuming Guesser is your model class
    guesser_state_dict = torch.load(guesser_load_path)
    model.load_state_dict(guesser_state_dict)
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.shape[0], -1)
            # call mask function on images
            images = mask(images)
            images = images.float()
            output = model(images)
            _, predicted = torch.max(output.data, 1)

            # Append true and predicted labels to calculate confusion matrix
            y_true.extend(labels.numpy())  # Assuming labels is a numpy array
            y_pred.extend(predicted.numpy())  # Assuming predicted is a numpy array

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate and print confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate accuracy
    accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
    print(f'Test Accuracy: {accuracy:.2f}')


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
    #count total sampels in trainloaser
    total_samples = len(train_loader.dataset)
    for i in range(1, nepochs):
        running_loss = 0
        for images, labels in train_loader:
            images = images.view(images.shape[0], -1)
            images = mask(images)
            images = images.float()
            model.train()
            model.optimizer.zero_grad()
            output = model(images)
            y_pred = []
            for y in labels:
                y = torch.Tensor(y).long()
                y_pred.append(y)
            labels = torch.Tensor(np.array(y_pred)).long()
            loss = model.criterion(output, labels)
            running_loss += loss.item()
            loss.backward()
            model.optimizer.step()

        print(f'Epoch: {i}, training loss: {running_loss / total_samples}')
        new_best_val_auc = val(model, val_loader, best_val_auc)
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
    guesser_filename = '../model_guesser/best_guesser.pth'
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
    X_tensor_test = torch.Tensor(X_test)
    y_tensor_test = torch.Tensor(y_test)  # Assuming y_data contains integers
    dataset_test = TensorDataset(X_tensor_test, y_tensor_test)
    data_loader_test = DataLoader(dataset_test, batch_size=FLAGS.batch_size, shuffle=True)
    test(data_loader_test, model.path_to_save)


if __name__ == "__main__":
    os.chdir(FLAGS.directory)
    main()
