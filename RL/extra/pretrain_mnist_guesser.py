# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:23:11 2020

@author: urixs
"""
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import count
from sklearn.metrics import confusion_matrix
import argparse
from collections import deque
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--save_dir",
                    type=str,
                    default='C:\\Users\kashann\\PycharmProjects\\adaptivFS\\pretrained_mnist_guesser_models',
                    help="Directory for saved models")
parser.add_argument("--hidden-dim",
                    type=int,
                    default=256,
                    help="Hidden dimension")
parser.add_argument("--lr",
                    type=float,
                    default=1e-4,
                    help="Learning rate")
parser.add_argument("--weight_decay",
                    type=float,
                    default=0.,
                    help="l_2 weight penalty")
parser.add_argument("--case",
                    type=int,
                    default=2,
                    help="Which data to use")
parser.add_argument("--val_trials_wo_im",
                    type=int,
                    default=50,
                    help="Number of validation trials without improvement")
parser.add_argument("--val_interval",
                    type=int,
                    default=1000,
                    help="Interval for calculating validation reward and saving model")

FLAGS = parser.parse_args(args=[])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Guesser(nn.Module):
    """
    implements a net that guesses the outcome given the state
    """
    
    def __init__(self, 
                 state_dim, 
                 hidden_dim=FLAGS.hidden_dim, 
                 num_classes=10):
        super(Guesser, self).__init__()
        
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.PReLU(), 
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.PReLU(),
        )
        
        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.PReLU(),
        )
        
        
        # output layer
        self.logits = nn.Linear(hidden_dim, num_classes)
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = torch.optim.Adam(self.parameters(), 
                                          weight_decay=FLAGS.weight_decay,
                                          lr=FLAGS.lr)


    def forward(self, x):
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        logits = self.logits(x)
        probs = F.softmax(logits, dim=1)
        
        return logits, probs   
    
    def _to_variable(self, x: np.ndarray) -> torch.Tensor:
        """torch.Variable syntax helper
        Args:
            x (np.ndarray): 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: torch variable
        """
        return torch.autograd.Variable(torch.Tensor(x))
    
def save_network(i_episode, guesser, acc=None):
    """ A function that saves the gesser params"""
    
    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)
    
    if i_episode == 'best':
        guesser_filename = 'best_guesser.pth'
    else:
        guesser_filename = '{}_{}_{:1.3f}.pth'.format(i_episode, 'guesser', acc)
        
    guesser_save_path = os.path.join(FLAGS.save_dir, guesser_filename)
    
    # save guesser
    if os.path.exists(guesser_save_path):
        os.remove(guesser_save_path)
    torch.save(guesser.cpu().state_dict(), guesser_save_path + '~')
    guesser.to(device=device)
    os.rename(guesser_save_path + '~', guesser_save_path)

def mask(x: np.array) -> np.array:
    """ A function that masks the input vector"""

    for i in range(int(len(x)/2)):
        # choose to mask in probability of 0.3
        if np.random.rand() < 0.3:
            x[i] = 0
            x[i + int(len(x) / 2)] = 0
    return x


def main():
    # Load data and randomly split to train, validation and test sets
    n_questions = 28 * 28

    # Initialize guesser
    guesser = Guesser(2 * n_questions)
    guesser.to(device=device)

    X_train, X_test, y_train, y_test = utils.load_mnist(case=FLAGS.case)

    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.33)

    if len(X_val > 10000):
        X_val = X_val[:10000]
        y_val = y_val[:10000]
    """ Main """
    #create train_loader and val_loader based on x_train, y_train

    # Delete models from earlier runs
    if os.path.exists(FLAGS.save_dir):
        shutil.rmtree(FLAGS.save_dir)
    
    # Reset counter
    val_trials_without_improvement = 0
    
    
    losses = deque(maxlen=100)
    best_val_acc = 0
    
    for i in count(1):

     patient = np.random.randint(X_train.shape[0])
     x = X_train[patient]
     x = np.concatenate([x, np.ones(n_questions)])
     #mask some features
     x= mask(x)
     guesser_input = guesser._to_variable(x.reshape(-1, 2 * n_questions))
     guesser_input = guesser_input.to(device=device)
     guesser.train(mode=False)
     logits, probs = guesser(guesser_input)
     y_true = y_train[patient]
     y = torch.Tensor([y_true]).long()
     y = y.to(device=device)
     guesser.optimizer.zero_grad()             
     guesser.train(mode=True)
     loss = guesser.criterion(logits, y) 
     losses.append(loss.item())       
     loss.backward()
     guesser.optimizer.step()
     
     if i % 100 == 0:
         print('Step: {}, loss={:1.3f}'.format(i, loss.item()))
     
        # COmpute performance on validation set and reset counter if necessary    
     if i % FLAGS.val_interval == 0:
        new_best_val_acc = val(i_episode=i, best_val_acc=best_val_acc,X_val=X_val,y_val=y_val,n_questions=n_questions,guesser=guesser)
        if new_best_val_acc > best_val_acc:
                    best_val_acc = new_best_val_acc
                    val_trials_without_improvement = 0
        else:
            val_trials_without_improvement += 1
            
    # check whether to stop training
        if val_trials_without_improvement == FLAGS.val_trials_wo_im:
            print('Did not achieve val acc improvement for {} trials, training is done.'.format(FLAGS.val_trials_wo_im))
            break
    test(i_episode=i, X_test=X_test, y_test=y_test, n_questions=n_questions, guesser=guesser)

def batch():
    # Load data and randomly split to train, validation and test sets
    n_questions = 28 * 28

    # Initialize guesser
    guesser = Guesser(2 * n_questions)
    guesser.to(device=device)

    X_train, X_test, y_train, y_test = utils.load_mnist(case=FLAGS.case)

    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.33)

    if len(X_val > 10000):
        X_val = X_val[:10000]
        y_val = y_val[:10000]
    """ Main """
    # create train_loader and val_loader based on x_train, y_train

    # Delete models from earlier runs
    if os.path.exists(FLAGS.save_dir):
        shutil.rmtree(FLAGS.save_dir)

    # Reset counter
    val_trials_without_improvement = 0

    losses = deque(maxlen=100)
    best_val_acc = 0

    # Set batch size
    batch_size = 32

    #
    # for i in count(20000):
    #     # Randomly select batch_size number of patients
    #     indices = np.random.choice(X_train.shape[0], batch_size, replace=False)
    #     batch_x = X_train[indices]
    #
    #     # Initialize an array to hold concatenated features for the batch
    #     batch_concatenated_x = np.empty((batch_size,  n_questions*2))
    #
    #     # Loop through batch samples to process them
    #     for idx, x in enumerate(batch_x):
    #         # Concatenate features and add additional ones (assuming n_questions additional features)
    #         x = np.concatenate([x, np.ones(n_questions)])
    #
    #         # Mask some features using the mask function
    #         x = mask(x)
    #
    #         # Append the modified x to the batch_concatenated_x array
    #         batch_concatenated_x[idx] = x
    #
    #     # Convert the batch data to PyTorch tensors
    #     batch_guesser_input = torch.from_numpy(batch_concatenated_x.reshape(-1, 2 * n_questions)).float()
    #     batch_y_true = torch.from_numpy(y_train[indices]).long()
    #
    #     # Move tensors to the device (assuming device is defined)
    #     batch_guesser_input = batch_guesser_input.to(device=device)
    #     batch_y_true = batch_y_true.to(device=device)
    #
    #     # Set the guesser model to evaluation mode
    #     guesser.train(mode=False)
    #
    #     # Forward pass
    #     logits, probs = guesser(batch_guesser_input)
    #
    #     # Calculate loss
    #     loss = guesser.criterion(logits, batch_y_true)
    #
    #     # Backpropagation
    #     guesser.optimizer.zero_grad()
    #     loss.backward()
    #     guesser.optimizer.step()
    #
    #     # Append loss to a list (losses)
    #     losses.append(loss.item())
    #     if i % 100 == 0:
    #         print('Step: {}, loss={:1.3f}'.format(i, loss.item()))
    #
    #     # COmpute performance on validation set and reset counter if necessary
    #     if i % FLAGS.val_interval == 0:
    #         new_best_val_acc = val(i_episode=i, best_val_acc=best_val_acc, X_val=X_val, y_val=y_val,
    #                                n_questions=n_questions, guesser=guesser)
    #         if new_best_val_acc > best_val_acc:
    #             best_val_acc = new_best_val_acc
    #             val_trials_without_improvement = 0
    #         else:
    #             val_trials_without_improvement += 1
    #
    #         # check whether to stop training
    #         if val_trials_without_improvement == FLAGS.val_trials_wo_im:
    #             print('Did not achieve val acc improvement for {} trials, training is done.'.format(
    #                 FLAGS.val_trials_wo_im))
    #             break
    test(X_test=X_test, y_test=y_test, n_questions=n_questions, guesser=guesser)


def val(i_episode : int,
        best_val_acc : float, X_val,y_val,n_questions,guesser) -> float:
    """ Computes performance on validation set """
    
    print('Running validation')
    y_hat_val = np.zeros(len(y_val))
    
    for i in range(len(X_val)):
        x = X_val[i]
        x = np.concatenate([x, np.ones(n_questions)])
        guesser_input = guesser._to_variable(x.reshape(-1, 2 * n_questions))
        guesser_input = guesser_input.to(device=device)
        guesser.train(mode=False)
        logits, probs = guesser(guesser_input)
        y_hat_val[i] = torch.argmax(probs).item()

    confmat = confusion_matrix(y_val,  y_hat_val)
    acc = np.sum(np.diag(confmat)) / len(y_val)
    #save_network(i_episode, acc)
    
    if acc > best_val_acc:
        print('New best Acc acheievd, saving best model')
        save_network( i_episode='best',guesser=guesser, acc=acc)
        # save_network(guesser,i_episode, acc)
        
        return acc
    
    else:
        return best_val_acc


def test(X_test, y_test, n_questions, guesser) -> float:
    """ Computes performance on validation set """

    print('Running test')
    y_hat_test = np.zeros(len(y_test))

    for i in range(len(X_test)):
        x = X_test[i]
        x = np.concatenate([x, np.ones(n_questions)])
        guesser_input = guesser._to_variable(x.reshape(-1, 2 * n_questions))
        guesser_input = guesser_input.to(device=device)
        guesser.train(mode=False)
        logits, probs = guesser(guesser_input)
        y_hat_test[i] = torch.argmax(probs).item()

    confmat = confusion_matrix(y_test, y_hat_test)
    acc = np.sum(np.diag(confmat)) / len(y_test)
    # save_network(i_episode, acc)
    print('Test accuracy: {}'.format(acc))



if __name__ == '__main__':
    batch()