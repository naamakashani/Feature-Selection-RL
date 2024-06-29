import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import utils_lstm as utils
from sklearn.model_selection import train_test_split
import numpy as np

# hyperparameters

sequence_length = 8
input_size = 1
hidden_size = 128
num_layers = 2
num_classes = 2
batch_size = 100
num_epochs = 200
learning_rate = 0.01


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, num_classes)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.path_to_save = os.path.join(os.getcwd(), '../lstm_model/model_guesser_lstm')
        self.X, self.y, self.question_names, self.features_size = utils.load_diabetes()

    def forward(self, X):
        hidden_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        cell_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        out, _ = self.lstm(X, (hidden_states, cell_states))
        out = self.output_layer(out[:, -1, :])
        return out


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


def val(model, val_dataloader, best_val_auc=0):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch, (images, labels) in enumerate(val_dataloader):
            images = images.float()
            images = images.reshape(-1, sequence_length, input_size)
            outputs = model(images)
            y_pred = []
            for y in labels:
                y = torch.Tensor(y).long()
                y_pred.append(y)
            labels = torch.Tensor(np.array(y_pred)).long()
            predicted = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print('Accuracy of the network on the test images: {} %'.format(100 * accuracy))
        if accuracy >= best_val_auc:
            save_model(model)
        return accuracy


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
            fraction = 0.3
            if (np.random.rand() < fraction):
                input[i] = 0
        return input
    else:
        for j in range(int(len(input))):
            for i in range(input[0].shape[0]):
                # fraction = np.random.uniform(0, 1)
                fraction = 0.3
                if (np.random.rand() < fraction):
                    input[j][i] = 0
        return input


def train(num_epochs, model, train_dataloader, data_loader_val):
    best_val_auc = 0
    total_step = len(train_dataloader)
    val_trials_without_improvement = 0
    max_val_trials_wo_im = 20
    for epoch in range(num_epochs):
        for batch, (images, labels) in enumerate(train_dataloader):
            images = images.float()
            images = images.reshape(-1, sequence_length, input_size)
            images = mask(images)
            outputs = model(images)
            y_pred = []
            for y in labels:
                y = torch.Tensor(y).long()
                y_pred.append(y)
            labels = torch.Tensor(np.array(y_pred)).long()
            loss = model.loss_function(outputs, labels)
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            if (batch + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, batch + 1, total_step, loss.item()))
        if epoch % 2 == 0:
            acuuracy = val(model, data_loader_val, best_val_auc)
            if acuuracy > best_val_auc:
                best_val_auc = acuuracy
                val_trials_without_improvement = 0
            else:
                val_trials_without_improvement += 1
            # check whether to stop training
            if val_trials_without_improvement == max_val_trials_wo_im:
                print('Did not achieve val AUC improvement for {} trials, training is done.'.format(
                    max_val_trials_wo_im))
                break


def test(test_loader, path_to_save):
    guesser_filename = 'best_guesser.pth'
    guesser_load_path = os.path.join(path_to_save, guesser_filename)
    model = LSTM()
    guesser_state_dict = torch.load(guesser_load_path)
    model.load_state_dict(guesser_state_dict)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(-1, sequence_length, input_size)
            images = images.float()
            outputs = model(images)
            y_pred = []
            for y in labels:
                y = torch.Tensor(y).long()
                y_pred.append(y)
            labels = torch.Tensor(np.array(y_pred)).long()
            predicted = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print('Accuracy of the network on the test images: {} %'.format(100 * accuracy))


def main():
    model = LSTM()

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
    data_loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

    # Convert data to PyTorch tensors
    X_tensor_val = torch.Tensor(X_val)
    y_tensor_val = torch.Tensor(y_val)  # Assuming y_data contains integers
    dataset_val = TensorDataset(X_tensor_val, y_tensor_val)
    data_loader_val = DataLoader(dataset_val, batch_size=32, shuffle=True)
    train(num_epochs, model, data_loader_train, data_loader_val)
    X_tensor_test = torch.Tensor(X_test)
    y_tensor_test = torch.Tensor(y_test)  # Assuming y_data contains integers
    dataset_test = TensorDataset(X_tensor_test, y_tensor_test)
    data_loader_test = DataLoader(dataset_test, batch_size=32, shuffle=True)

    test(data_loader_test, model.path_to_save)


if __name__ == "__main__":
    os.chdir("/RL/lstm_model\\")
    main()
