# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 21:58:47 2020

This model trains a MLP model on x, another mlp model on t, 
and then a meta model on the outputs of the previous two

@author: urixs
"""


import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
import shap
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import date
import os
import shutil
from itertools import count
import nltk
import pickle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

'''  Config params '''
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--save_dir_models",
                    type=str,
                    default=r'C:\Users\kashann\PycharmProjects\choiceMira\codeChoice\mlp_models',
                    help="Directory for saved models")
parser.add_argument("--save_dir_shap",
                    type=str,
                    default=r'C:\Users\kashann\PycharmProjects\choiceMira\codeChoice\shap_plots',
                    help="Directory for shap plots")
parser.add_argument("--load_dir",
                    type=str,
                    default=r'C:\Users\kashann\PycharmProjects\choiceMira\codeChoice\word_embeddings',
                    help="Directory for saved models")
parser.add_argument("--load_pretrained_embeddings",
                    type=int,
                    default=0,
                    help="Whether to load cbow-pretrained word embeddings")
parser.add_argument("--inputs",
                    type=str,
                    default='both',
                    help="x | t | both")
parser.add_argument("--outcomes",
                    type=str,
                    default='dtd',
                    help="both | dtd | readmission")
parser.add_argument("--embedding_dim",
                    type=int,
                    default='64',
                    help="Embedding dimension")
parser.add_argument("--seq_length",
                    type=int,
                    default=200,
                    help="Sequence length - unused")
parser.add_argument("--hidden_dim",
                    type=int,
                    default=128,
                    help="Hidden dimension")
parser.add_argument("--dropout_prob",
                    type=float,
                    default=.5,
                    help="Dropout probability")
parser.add_argument("--batch_size",
                    type=int,
                    default=64,
                    help="batch_size")
parser.add_argument("--lr",
                    type=float,
                    default=1e-3,
                    help="Learning rate")
parser.add_argument("--min_lr",
                    type=float,
                    default=1e-5,
                    help="Minimal learning rate")
parser.add_argument("--decay_step_size",
                    type=int,
                    default=10,
                    help="LR decay step size")
parser.add_argument("--lr_decay_factor",
                    type=float,
                    default=0.5,
                    help="LR decay factor")
parser.add_argument("--cyclic_lr",
                    type=int,
                    default=0,
                    help="whether to use cyclic learning rate")
parser.add_argument("--weight_decay",
                    type=float,
                    default=1e-3,
                    help="l_2 weight penalty")
parser.add_argument("--val_interval",
                    type=int,
                    default=1,
                    help="Do validation every this number of epochs")
parser.add_argument("--max_vals_wo_improvement",
                    type=int,
                    default=3,
                    help="number of val steps without improvement before stop training")
parser.add_argument("--vals_to_unfreeze_embedding",
                    type=int,
                    default=1,
                    help="number of val steps before training embedding weights")
parser.add_argument("--meta_n_epochs",
                    type=int,
                    default=10,
                    help="number of epochs for the meta model")


FLAGS = parser.parse_args(args=[])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N_TRIALS = 5

if os.path.exists(FLAGS.save_dir_shap):
    shutil.rmtree(FLAGS.save_dir_shap)
os.makedirs(FLAGS.save_dir_shap)

if os.path.exists(FLAGS.save_dir_models):
    shutil.rmtree(FLAGS.save_dir_models)
os.makedirs(FLAGS.save_dir_models)


''' ------- Helper functions for text ------- ''' 

def load_text_objects():
    
    # load word2idx
    word2idx_load_path = os.path.join(FLAGS.load_dir, 'word2idx.pkl')
    with open(word2idx_load_path, 'rb') as f:
        word2idx = pickle.load(f)
        
    # load idx2word
    idx2word_load_path = os.path.join(FLAGS.load_dir, 'idx2word.pkl')
    with open(idx2word_load_path, 'rb') as f:
        idx2word = pickle.load(f)
        
   # load vocab
    vocab_load_path = os.path.join(FLAGS.load_dir, 'vocab.pkl')
    with open(vocab_load_path, 'rb') as f:
        vocab = pickle.load(f)
    vocab.insert(0, 'pad')
        
    # load embeddings
    embedding_filename = 'best_embedding.pth'
    cbow_load_path = os.path.join(FLAGS.load_dir, embedding_filename)
    cbow_state_dict = torch.load(cbow_load_path)
    embedding_weights = cbow_state_dict['embeddings.weight']
    embedding_weights = np.insert(embedding_weights, 0, np.zeros(FLAGS.embedding_dim), axis=0)
    return word2idx, idx2word, vocab, embedding_weights

def create_text_objects(tokenized_corpus):
    
    vocab = []
    for sentence in tokenized_corpus:
        for token in sentence:
            if token not in vocab:
                vocab.append(token)    
    vocab.insert(0, 'pad')

    
    print('Created Vocabulary, size: {}'.format(len(vocab)))
    
    word2idx = {w: idx for (idx, w) in enumerate(vocab)}   
    idx2word = {idx: w for (idx, w) in enumerate(vocab)}
    
    return word2idx, idx2word, vocab

def tokenize_sentence(sentence):
    """ Removes punctuation and splits to words """
    
    if type(sentence) == str:
        words = nltk.wordpunct_tokenize(sentence)
        sentence_tokens = [word for word in words if word.isalnum()]
    else:
        sentence_tokens = []
    return sentence_tokens

def tokenize_corpus(corpus):
    """ tokenizes each sentence in the corpus """
    
    tokenized_corpus = [tokenize_sentence(sentence) for sentence in corpus]
    return tokenized_corpus


def create_vocab(tokenized_corpus):
    """ All unique tokens from corpus """
    
    vocabulary = []
    for sentence in tokenized_corpus:
        for token in sentence:
            if token not in vocabulary:
                vocabulary.append(token)
    print('Created Vocabulary, size: {}'.format(len(vocabulary)))
    return vocabulary

def get_sentence_idxs(tokenized_sentence):
    """ returns list of indices of the context words """ 
    
    idxs = [word2idx[w] for w in tokenized_sentence]
    return idxs

def pad_corpus(idxs_corpus, seq_length=FLAGS.seq_length):
    ''' Return features of idxs_corpus, where each sentence padded with zeros or truncated.
    '''
    padded_corpus = np.zeros((len(idxs_corpus), seq_length), dtype=int)
    
    for i, sentence_idxs in enumerate(idxs_corpus):
        sentence_len = len(sentence_idxs)
        
        if sentence_len <= seq_length:
            zeros = list(np.zeros(seq_length - sentence_len))
            new = zeros + sentence_idxs
        elif sentence_len > seq_length:
            new = sentence_idxs[0 : seq_length]
        
        padded_corpus[i, :] = np.array(new)
    
    return padded_corpus

''' ------- Load data -------'''

outcomes = pd.read_pickle(r'C:\Users\kashann\PycharmProjects\choiceMira\codeChoice\data_rl\outcomes.pkl')
n_outcomes = outcomes.shape[1]
outcome_names = outcomes.columns
Y = outcomes.to_numpy()
if FLAGS.outcomes == 'dtd':
    dtd_indices = [0]#[i for i, name in enumerate(outcome_names) if 'dtd' in name]
    Y = Y[:, dtd_indices]
    n_outcomes = len(dtd_indices)
    outcome_names = outcome_names[dtd_indices]
elif FLAGS.outcomes == 'readmission':
    readmission_indices = [4]#[i for i, name in enumerate(outcome_names) if 'readmission' in name]
    Y = Y[:, readmission_indices]
    n_outcomes = len(readmission_indices)
    outcome_names = outcome_names[readmission_indices]
elif FLAGS.outcomes == 'both':
    n_outcomes = 2
    outcome_names = outcome_names[[0, 4]]
    Y = Y[:, [0, 4]]
elif FLAGS.outcomes == 'all':
    pass
else:
    print ('FLAGS.outcomes not recognized')
X_pd = pd.read_pickle(r'C:\Users\kashann\PycharmProjects\choiceMira\codeChoice\data_rl\preprocessed_X.pkl')
X = X_pd.to_numpy()
scaler = StandardScaler()
#X = scaler.fit_transform(X) #Do not scale if using shap
Data = pd.read_csv(r'C:\Users\kashann\PycharmProjects\choiceMira\codeChoice\data_rl\new_data_apr22.csv')
admission_date = pd.to_datetime(Data['Reference Event-Visit Start Date']) 

X = X.astype('float32')
Y = Y.astype('int')

Text = pd.read_pickle(r'C:\Users\kashann\PycharmProjects\choiceMira\codeChoice\data_rl\text.pkl')['Free-text-all-exam-res']#['ct_general']
tokenized_corpus = tokenize_corpus(Text)

if FLAGS.load_pretrained_embeddings:
    word2idx, idx2word, vocab, embedding_weights = load_text_objects()
else:
    word2idx, idx2word, vocab = create_text_objects(tokenized_corpus)
    
vocab_size = len(vocab)    

# Converting corpus to indices
idxs_corpus = [get_sentence_idxs(tokenized_sentence) for tokenized_sentence in tokenized_corpus]

# Arrange idxs_corpus in a multihot way
multihot_corpus = np.zeros([len(idxs_corpus), vocab_size]).astype('float32')
for i in range(len(idxs_corpus)):
    for j in range(len(idxs_corpus[i])):
       multihot_corpus[i, idxs_corpus[i][j]] += 1. 
   
''' ------- Divide data to train and test ------- '''    
n = len(X)
perm = np.random.permutation(n)
test_inds = perm[-int(n * .2):]
val_inds = perm[-int(n * .325) : -int(n * .2)]
train_inds = perm[: -int(n * .325)]
n_train = len(train_inds)
n_val = len(val_inds)
n_test = len(test_inds)

X_train = X[train_inds]
T_train = multihot_corpus[train_inds]
Y_train = Y[train_inds]
X_val   = X[val_inds]
T_val   = multihot_corpus[val_inds]
Y_val   = Y[val_inds]
X_test  = X[test_inds]
T_test  = multihot_corpus[test_inds]
Y_test  = Y[test_inds]       
       

x_train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))       
t_train_data = TensorDataset(torch.from_numpy(T_train), torch.from_numpy(Y_train))
x_val_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
t_val_data = TensorDataset(torch.from_numpy(T_val), torch.from_numpy(Y_val))
x_test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
t_test_data = TensorDataset(torch.from_numpy(T_test), torch.from_numpy(Y_test))
        

x_train_loader = DataLoader(x_train_data, shuffle=True, batch_size=FLAGS.batch_size)
t_train_loader = DataLoader(t_train_data, shuffle=True, batch_size=FLAGS.batch_size)
x_val_loader = DataLoader(x_val_data, shuffle=False, batch_size=FLAGS.batch_size)
t_val_loader = DataLoader(t_val_data, shuffle=False, batch_size=FLAGS.batch_size)
x_test_loader = DataLoader(x_test_data, shuffle=False, batch_size=FLAGS.batch_size)        
t_test_loader = DataLoader(t_test_data, shuffle=False, batch_size=FLAGS.batch_size)        
  

''' ------- Compute class weights ------- '''
   
class_weights = []
for i in range(n_outcomes):
    y_train = Y_train[:, i]

    class_1_prop = np.sum(y_train) / len(y_train)
    class_0_prop = 1 - class_1_prop
    outcome_class_weights = [1 / class_0_prop, 1 / class_1_prop]
    outcome_class_weights /= np.sum(outcome_class_weights)
    outcome_class_weights = torch.Tensor(outcome_class_weights)
    class_weights.append(outcome_class_weights)
   
''' ------- Models ------- '''

class x_MLP(nn.Module):
    """
    The MLP model that will be used to perform outcome prediction.
    """ 

    def __init__(self, input_dim, hidden_dim, output_dim, drop_prob=0.25):
        """
        Initialize the model by setting up the layers.
        """
        super(x_MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        
        # X Hidden layers
        self.layer1 = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_prob)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_prob)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_prob)
        )    
        
        self.layer4 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_prob)
        )
        
        self.layer5 = nn.Sequential(
            nn.Linear(self.hidden_dim, output_dim),
            nn.Sigmoid(),
        )    
                
    def forward(self, x):
        """
        Perform a forward pass of our model on some input
        """
        
        x = self.layer1(x)
        x = x + self.layer2(x)
        x = x + self.layer3(x)
        x = x + self.layer4(x)
        x = self.layer5(x)
        
        return x
    
class t_MLP(nn.Module):
    """
    The MLP model that will be used to perform outcome prediction.
    """ 

    def __init__(self, hidden_dim, output_dim, drop_prob=0.25):
        """
        Initialize the model by setting up the layers.
        """
        super(t_MLP, self).__init__()

        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.embedding_dim = FLAGS.embedding_dim
        
        # Embedding layer
        self.embedding = nn.Linear(vocab_size, self.embedding_dim, bias=True)
        if FLAGS.load_pretrained_embeddings:
            self.embedding.weight = nn.Parameter(embedding_weights.T)
        else:
            torch.nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0)
        
        self.layer1 = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_prob)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_prob)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_prob)
        )    
        
        self.layer4 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_prob)
        )
        
        
        self.layer5 = nn.Sequential(
            nn.Linear(self.hidden_dim, output_dim),
            nn.Sigmoid(),
        )    
                
    def forward(self, t):
        """
        Perform a forward pass of our model on some input
        """

        
        t = self.embedding(t)
        t = self.layer1(t)
        t = t + self.layer2(t)
        t = t + self.layer3(t)
        t = t + self.layer4(t)
        t = self.layer5(t)
        
        return t
    
class meta_MLP(nn.Module):
    """
    The meta MLP model that will be used to perform outcome prediction.
    """ 

    def __init__(self, hidden_dim, output_dim, drop_prob=0.25):
        """
        Initialize the model by setting up the layers.
        """
        super(meta_MLP, self).__init__()

        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.input_dim = 2 *  n_outcomes
          
        self.layer1 = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_prob)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(self.hidden_dim, output_dim),
            nn.Sigmoid()
        )
                 
    def forward(self, x):
        """
        Perform a forward pass of our model on some input
        """
   
        x = self.layer1(x)
        x = self.layer2(x)
        
        return x
    
x_model = x_MLP(input_dim=X.shape[1],
                hidden_dim=FLAGS.hidden_dim, 
                output_dim=n_outcomes,
                drop_prob=FLAGS.dropout_prob).to(device=device)

t_model = t_MLP(hidden_dim=FLAGS.hidden_dim, 
                output_dim=n_outcomes,
                drop_prob=FLAGS.dropout_prob).to(device=device)

meta_model = meta_MLP(hidden_dim=FLAGS.hidden_dim, 
                      output_dim=n_outcomes,
                      drop_prob=FLAGS.dropout_prob).to(device=device)


''' ------- LR and optimizer ------- '''

if FLAGS.cyclic_lr == 1:
    
    x_optim = torch.optim.RMSprop(x_model.parameters(), 
                                  lr=FLAGS.lr, 
                                  weight_decay=FLAGS.weight_decay)   
    
    x_scheduler = lr_scheduler.CyclicLR(x_optim, 
                                        base_lr = FLAGS.min_lr,
                                        max_lr = FLAGS.lr,
                                        step_size_up=100)
    
    t_optim = torch.optim.RMSprop(t_model.parameters(), 
                                  lr=FLAGS.lr, 
                                  weight_decay=FLAGS.weight_decay)   
    
    t_scheduler = lr_scheduler.CyclicLR(t_optim, 
                                        base_lr = FLAGS.min_lr,
                                        max_lr = FLAGS.lr,
                                        step_size_up=100)
    
    m_optim = torch.optim.RMSprop(meta_model.parameters(), 
                                  lr=FLAGS.lr, 
                                  weight_decay=FLAGS.weight_decay)   
    
    m_scheduler = lr_scheduler.CyclicLR(m_optim, 
                                        base_lr = FLAGS.min_lr,
                                        max_lr = FLAGS.lr,
                                        step_size_up=100)
        
else:
    x_optim = torch.optim.Adam(x_model.parameters(), 
                               lr=FLAGS.lr, 
                               weight_decay=FLAGS.weight_decay)  
    
    t_optim = torch.optim.Adam(t_model.parameters(), 
                               lr=FLAGS.lr, 
                               weight_decay=FLAGS.weight_decay)  
    
    m_optim = torch.optim.Adam(meta_model.parameters(), 
                               lr=FLAGS.lr, 
                               weight_decay=FLAGS.weight_decay)  
 
    def lambda_rule(i_episode) -> float:
        """ stepwise learning rate calculator """
        exponent = int(np.floor((i_episode + 1) / FLAGS.decay_step_size))
        return np.power(FLAGS.lr_decay_factor, exponent)
    
    x_scheduler = lr_scheduler.LambdaLR(x_optim, 
                                        lr_lambda=lambda_rule) 
    
    t_scheduler = lr_scheduler.LambdaLR(t_optim, 
                                        lr_lambda=lambda_rule) 
    
    m_scheduler = lr_scheduler.LambdaLR(m_optim, 
                                        lr_lambda=lambda_rule) 
    
    def x_update_lr():
        """ Learning rate updater """
        
        x_scheduler.step()
        lr = x_optim.param_groups[0]['lr']
        if lr < FLAGS.min_lr:
            x_optim.param_groups[0]['lr'] = FLAGS.min_lr
            lr = x_optim.param_groups[0]['lr']
        print('Learning rate = %.7f' % lr) 
            
    def t_update_lr():
        """ Learning rate updater """
        
        t_scheduler.step()
        lr = t_optim.param_groups[0]['lr']
        if lr < FLAGS.min_lr:
            t_optim.param_groups[0]['lr'] = FLAGS.min_lr
            lr = t_optim.param_groups[0]['lr']
        print('Learning rate = %.7f' % lr) 
        
    def m_update_lr():
        """ Learning rate updater """
        
        m_scheduler.step()
        lr = m_optim.param_groups[0]['lr']
        if lr < FLAGS.min_lr:
            m_optim.param_groups[0]['lr'] = FLAGS.min_lr
            lr = m_optim.param_groups[0]['lr']
        print('Learning rate = %.7f' % lr) 
            
''' ------- Loss criterions ------- '''            
  
x_criterions = []
for i in range(n_outcomes):
    x_criterions.append(nn.BCELoss())    
    
t_criterions = []
for i in range(n_outcomes):
    t_criterions.append(nn.BCELoss())  
    
m_criterions = []
for i in range(n_outcomes):
    m_criterions.append(nn.BCELoss()) 
    
''' ------- Train, val and test procedures ------- '''            

def x_train_step(batch_x, batch_y):
    
    x_model.train(True)
    
    batch_x = batch_x.to(device=device)
    batch_y = batch_y.to(device=device)
                
    batch_output = x_model(batch_x)
    
    x_model.zero_grad()
    
    loss = 0
    for i in range(n_outcomes):
        loss += x_criterions[i](batch_output[:, i], batch_y[:, i].float())
    
    loss.backward()
    x_optim.step()
    
    return batch_output, loss.item()    

def t_train_step(batch_t, batch_y):
    
    t_model.train(True)
    
    batch_t = batch_t.to(device=device)
    batch_y = batch_y.to(device=device)
                
    batch_output = t_model(batch_t)
    
    t_model.zero_grad()
    
    loss = 0
    for i in range(n_outcomes):
        loss += t_criterions[i](batch_output[:, i], batch_y[:, i].float())
    
    loss.backward()
    t_optim.step()
    
    return batch_output, loss.item()

def meta_train_step(batch_x, batch_y):
    
    meta_model.train(True)
    
    batch_x = batch_x.to(device=device)
    batch_y = batch_y.to(device=device)
                
    batch_output = meta_model(batch_x)
    
    meta_model.zero_grad()
    
    loss = 0
    for i in range(n_outcomes):
        loss += m_criterions[i](batch_output[:, i], batch_y[:, i].float())
    
    loss.backward()
    m_optim.step()
    
    return batch_output, loss.item()

def x_run_validation():
        
    x_model.train(False)
    
    print('Running validation')
    losses = []
    probs = []

    for batch_x, batch_y in x_val_loader:
                    
        batch_x = batch_x.to(device=device)
        batch_y = batch_y.to(device=device)
                    
        # set criterion_weights
        for i in range(n_outcomes):
            x_criterions[i].weight = class_weights[i][batch_y[:, i].cpu().numpy()].to(device=device)    
            
        batch_output = x_model(batch_x)            
        
        loss = 0
        for i in range(n_outcomes):
            loss += x_criterions[i](batch_output[:, i], batch_y[:, i].float())
        
        losses.append(loss.item())  
        probs.append(batch_output.cpu().detach().numpy())
    
    probs = np.concatenate(probs)
    avg_auc = 0
    for i in range(n_outcomes):
        avg_auc += roc_auc_score(Y_val[:, i],  probs[:, i])
    avg_auc /= n_outcomes
        
    return np.mean(losses), avg_auc, probs

def t_run_validation():
        
    t_model.train(False)
    
    print('Running validation')
    losses = []
    probs = []

    for batch_t, batch_y in t_val_loader:
                    
        batch_t = batch_t.to(device=device)
        batch_y = batch_y.to(device=device)
                    
        # set criterion_weights
        for i in range(n_outcomes):
            t_criterions[i].weight = class_weights[i][batch_y[:, i].cpu().numpy()].to(device=device)    
            
        batch_output = t_model(batch_t)            
        
        loss = 0
        for i in range(n_outcomes):
            loss += t_criterions[i](batch_output[:, i], batch_y[:, i].float())
        
        losses.append(loss.item())  
        probs.append(batch_output.cpu().detach().numpy())
    
    probs = np.concatenate(probs)
    avg_auc = 0
    for i in range(n_outcomes):
        avg_auc += roc_auc_score(Y_val[:, i],  probs[:, i])
    avg_auc /= n_outcomes
        
    return np.mean(losses), avg_auc, probs

def x_train():
    
    best_val_loss = 1000
    eps = 1e-6
    epoch_counter = 0    
     
    for epoch in count(1):
        
        step = 0
       
        losses = []
     
        for batch_x, batch_y in x_train_loader:
            
            # set criterion_weights
            for i in range(n_outcomes):
                x_criterions[i].weight = class_weights[i][batch_y[:, i].cpu().numpy()].to(device=device)    
           
            batch_output, loss = x_train_step(batch_x, batch_y)
            
            losses.append(loss)
        
            if step % 1000 == 0:
                print('Epoch: {}, Step: {}, loss: {:.3f}'.format(epoch + 1,
                                                                 step,
                                                                 np.mean(losses)))
                print('Learning rate = %.7f' % x_optim.param_groups[0]['lr'])
                      
            step += 1
       
        if epoch % FLAGS.val_interval == 0:
            val_loss, val_auc, _ = x_run_validation()
            if val_loss < best_val_loss - eps:
                best_val_loss = val_loss
                epoch_counter = 0
                print('New best val loss acheievd, saving best model')
                save_model(epoch, val_loss)
                save_model(epoch='best', modality='x')
            else:
                epoch_counter += 1
            print('End of epoch: {}, val loss: {:.3f}, val_auc: {:.3f}, counter: {}'.format(epoch + 1, 
                                                                                            val_loss, 
                                                                                            val_auc, 
                                                                                            epoch_counter))
            
        if epoch_counter == FLAGS.max_vals_wo_improvement:
            break
         
        x_update_lr()
        
def t_train():
    
    best_val_loss = 1000
    eps = 1e-6
    epoch_counter = 0    
     
    for epoch in count(1):
        
        step = 0
       
        losses = []
     
        for batch_t, batch_y in t_train_loader:
            
            # set criterion_weights
            for i in range(n_outcomes):
                t_criterions[i].weight = class_weights[i][batch_y[:, i].cpu().numpy()].to(device=device)    
           
            batch_output, loss = t_train_step(batch_t, batch_y)
            
            losses.append(loss)
        
            if step % 1000 == 0:
                print('Epoch: {}, Step: {}, loss: {:.3f}'.format(epoch + 1, 
                                                                 step, 
                                                                 np.mean(losses)))
                print('Learning rate = %.7f' % x_optim.param_groups[0]['lr']) 
                      
            step += 1
       
        if epoch % FLAGS.val_interval == 0:
            val_loss, val_auc, _ = t_run_validation()
            if val_loss < best_val_loss - eps:
                best_val_loss = val_loss
                epoch_counter = 0
                print('New best val loss acheievd, saving best model')
                save_model(epoch, val_loss)
                save_model(epoch='best', modality='t')
            else:
                epoch_counter += 1
            print('End of epoch: {}, val loss: {:.3f}, val_auc: {:.3f}, counter: {}'.format(epoch + 1, 
                                                                                            val_loss, 
                                                                                            val_auc, 
                                                                                            epoch_counter))
            
        if epoch_counter == FLAGS.max_vals_wo_improvement:
            break
         
        t_update_lr()
         
    print('Finished training')
    
def train_meta(x_probs, t_probs):
    
    probs = np.concatenate([x_probs, t_probs], axis=1)
    meta_train_dataset = TensorDataset(torch.from_numpy(probs), torch.from_numpy(Y_val))       
    meta_train_loader = DataLoader(meta_train_dataset, shuffle=True, batch_size=FLAGS.batch_size)

     
    for epoch in range(FLAGS.meta_n_epochs):
        
        step = 0
        losses = []
     
        for batch_x, batch_y in meta_train_loader:
            
            # set criterion_weights
            for i in range(n_outcomes):
                m_criterions[i].weight = class_weights[i][batch_y[:, i].cpu().numpy()].to(device=device)    
           
            batch_output, loss = meta_train_step(batch_x, batch_y)
            
            losses.append(loss)
        
            if step % 1000 == 0:
                print('Epoch: {}, Step: {}, loss: {:.3f}'.format(epoch + 1, 
                                                                 step, 
                                                                 np.mean(losses)))
                print('Learning rate = %.7f' % m_optim.param_groups[0]['lr']) 

        x_update_lr()
        
    save_model(epoch='best', modality='meta')

def x_run_inference():
        
    print('Running inference')
    
    # load best performing models
    print('Loading best model')
    x_model = load_model(epoch='best', modality='x')
    
    x_model.train(False)
    
    probs = []
          
    for batch_x, batch_y in x_test_loader:
                            
        batch_x = batch_x.to(device=device)
        batch_y = batch_y.to(device=device)
                    
        batch_output = x_model(batch_x)            
        probs.append(batch_output.cpu().detach().numpy())
    
    probs = np.concatenate(probs)
    aucs = []
    for i in range(n_outcomes):
        aucs.append(roc_auc_score(Y_test[:, i],  probs[:, i]))
    avg_auc = np.mean(aucs) 

    return probs, aucs, avg_auc

def t_run_inference():
        
    print('Running inference')
    
    # load best performing models
    print('Loading best model')
    t_model = load_model (epoch='best', modality='t')
    
    t_model.train(False)
    
    probs = []
          
    for batch_t, batch_y in t_test_loader:
                            
        batch_t = batch_t.to(device=device)
        batch_y = batch_y.to(device=device)
                    
        batch_output = t_model(batch_t)            
        probs.append(batch_output.cpu().detach().numpy())
    
    probs = np.concatenate(probs)
    aucs = []
    for i in range(n_outcomes):
        aucs.append(roc_auc_score(Y_test[:, i],  probs[:, i]))
    avg_auc = np.mean(aucs) 

    return probs, aucs, avg_auc

def meta_run_inference(x_probs, t_probs):
        
    print('Running inference')
    
    # load best performing models
    print('Loading best model')
    meta_model = load_model (epoch='best', modality='meta')
    
    meta_model.train(False)
    
    probs = []
    
    input_probs = np.concatenate([x_probs, t_probs], axis=1)
    meta_test_dataset = TensorDataset(torch.from_numpy(input_probs), torch.from_numpy(Y_test))       
    meta_test_loader = DataLoader(meta_test_dataset, shuffle=False, batch_size=FLAGS.batch_size)

          
    for batch_x, batch_y in meta_test_loader:
                            
        batch_x = batch_x.to(device=device)
        batch_y = batch_y.to(device=device)
                    
        batch_output = meta_model(batch_x)            
        probs.append(batch_output.cpu().detach().numpy())
    
    probs = np.concatenate(probs)
    aucs = []
    for i in range(n_outcomes):
        aucs.append(roc_auc_score(Y_test[:, i],  probs[:, i]))
    avg_auc = np.mean(aucs) 

    return probs, aucs, avg_auc


''' ------- Save and load models ------- '''     
    
def save_model(epoch: int, 
               val_loss=None,
               modality='x') -> None:
    """ A method to save model params"""
    if not os.path.exists(FLAGS.save_dir_models):
        os.makedirs(FLAGS.save_dir_models)
    
    if epoch == 'best':
        model_filename = 'best_{}_model.pth'.format(modality)
    else:
        model_filename = '{}_{}_{}_{:1.3f}.pth'.format(epoch, modality, 'model', val_loss)
        
    model_save_path = os.path.join(FLAGS.save_dir_models, model_filename)
    
    if os.path.exists(model_save_path):
        os.remove(model_save_path)
    
    if modality == 'x':
        torch.save(x_model.cpu().state_dict(), model_save_path + '~')
        x_model.to(device=device)
    elif modality == 't':
        torch.save(t_model.cpu().state_dict(), model_save_path + '~')
        t_model.to(device=device)
    elif modality == 'meta':
        torch.save(meta_model.cpu().state_dict(), model_save_path + '~')
        meta_model.to(device=device)
    os.rename(model_save_path + '~', model_save_path)
            
def load_model(epoch: int, 
               val_loss=None,
               modality='x') -> None:
    """ A method to load parameters of saved model"""
    if epoch == 'best':
        model_filename = 'best_{}_model.pth'.format(modality)
    else:
        model_filename = '{}_{}_{}_{:1.3f}.pth'.format(epoch, modality, 'model', val_loss)
        
    model_load_path = os.path.join(FLAGS.save_dir_models, model_filename)
    
    # load model
    if modality == 'x':
        model= x_MLP(input_dim=X.shape[1],
                     hidden_dim=FLAGS.hidden_dim, 
                     output_dim = n_outcomes,
                     drop_prob=FLAGS.dropout_prob)
    
    elif modality == 't':
        model = t_MLP(hidden_dim=FLAGS.hidden_dim, 
                      output_dim=n_outcomes,
                      drop_prob=FLAGS.dropout_prob).to(device=device)
    
    elif modality == 'meta':        
        model = meta_MLP(hidden_dim=FLAGS.hidden_dim, 
                         output_dim=n_outcomes,
                         drop_prob=FLAGS.dropout_prob).to(device=device)
        
    model_state_dict = torch.load(model_load_path)
    model.load_state_dict(model_state_dict)
    model.to(device=device)
    
    return model
      
''' ------- Model interpretation ------- '''         
    
def interpret_x_model():
    
    train_sample_inds = np.random.randint(n_train, size=100)
    test_sample_inds = np.random.randint(n_test, size=100)
    print('Computing Shap values')
    shap.initjs
    z_train = torch.Tensor(X_train[train_sample_inds]).to(device=device)
    z_test = torch.Tensor(X_test[test_sample_inds]).to(device=device)
    explainer = shap.DeepExplainer(x_model, z_train)
    shap_values = explainer.shap_values(z_test)
    if n_outcomes == 1:
        shap_values = [shap_values]
    
    # Summarize the effects of all x features
    Z_x_pd = pd.DataFrame(X_test[test_sample_inds])
    Z_x_pd.columns = X_pd.columns
    for i in range(n_outcomes):    
        outcome_name = outcome_names[i]
        plt.figure()
        fig = shap.summary_plot(shap_values[i], 
                                Z_x_pd, plot_size=(20, 10), 
                                title=outcome_name, 
                                show=False)
        ax = plt.gca()
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        fig_filename = '{}_x.png'.format(outcome_name)
        plt.savefig(os.path.join(FLAGS.save_dir_shap,  fig_filename))
        
    # Example of case study - patient #5, outcome #0
    # x features
    patient_num = 50
    outcome = 0
    shap.force_plot(explainer.expected_value[outcome], 
                    shap_values[outcome][patient_num], 
                    Z_x_pd.iloc[patient_num, :],
                    matplotlib=True)
        # red (blue) features: push prediction higher (lower)


def interpret_t_model():
    
    train_sample_inds = np.random.randint(n_train, size=100)
    test_sample_inds = np.random.randint(n_test, size=100)
    print('Computing Shap values')
    shap.initjs
    z_train = torch.Tensor(T_train[train_sample_inds])
    z_test = torch.Tensor(T_test[test_sample_inds])
    explainer = shap.DeepExplainer(t_model, z_train)
    shap_values = explainer.shap_values(z_test)
    if n_outcomes == 1:
        shap_values = [shap_values]
        
    # Summarize the effects of all words  
    rvocab = ['pad']
    for i in range(1, len(vocab)):
       rvocab.append(vocab[i][::-1]) 
    Z_t_pd = pd.DataFrame(T_test[test_sample_inds])
    Z_t_pd.columns = rvocab
    
    for i in range(n_outcomes):    
        outcome_name = outcome_names[i]
        
        plt.figure()
        fig = shap.summary_plot(shap_values[i], 
                                Z_t_pd, 
                                title=outcome_name,
                                plot_size=(20, 10), 
                                show=False)
        ax = plt.gca()
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        fig_filename = '{}_t.png'.format(outcome_name)
        plt.savefig(os.path.join(FLAGS.save_dir_shap,  fig_filename))
        
    # Example of case study - patient #5, outcome #0
    # words
    outcome = 0
    patient_num = 5
    shap.force_plot(explainer.expected_value[outcome], 
                shap_values[outcome][patient_num], 
                Z_t_pd.iloc[patient_num, :],
                matplotlib=True)
    # Note: the output value is the plots are not the real model output value, as the other input is missing
    
    
''' ------- Results display ------- '''  

def display_results(test_aucs_x, test_aucs_t, test_aucs_meta):
    today = date.today()   
    print('{} results:'.format(today.strftime('%B %d, %Y')))
    #print('Summary of mean(std) based on {} independent trials for each outcome:'.format(N_TRIALS))
    print('Outcome AUC scores (x):')
    for i, outcome_name in enumerate(outcome_names):
        print('Outcome: {}: {}, auc: {:.3f}'.format(i + 1, 
                                                      outcome_name, 
                                                      test_aucs_x[i])) 
    print('Outcome AUC scores (t):')
    for i, outcome_name in enumerate(outcome_names):
        print('Outcome: {}: {}, auc: {:.3f}'.format(i + 1, 
                                                      outcome_name, 
                                                      test_aucs_t[i])) 
        
    print('Outcome AUC scores (meta):')
    for i, outcome_name in enumerate(outcome_names):
        print('Outcome: {}: {}, auc: {:.3f}'.format(i + 1, 
                                                      outcome_name, 
                                                      test_aucs_meta[i])) 

''' ------- Main ------- ''' 

def main():
    
    x_train()
    x_test_probs, x_test_aucs, _ = x_run_inference()
    t_train()
    t_test_probs, t_test_aucs, _ = t_run_inference()
    _, _, x_val_probs = x_run_validation()
    _, _, t_val_probs = t_run_validation()
    train_meta(x_val_probs, t_val_probs)
    meta_test_probs, meta_test_aucs, _ = meta_run_inference(x_test_probs, 
                                                            t_test_probs)
    display_results(x_test_aucs, t_test_aucs, meta_test_aucs)
    interpret_x_model()
    interpret_t_model()
    
if __name__ == '__main__':
    os.chdir("C:\\Users\\kashann\\PycharmProjects\\choiceMira\\codeChoice")
    main()
    
'''
# 1
July 13, 2020 results:
Outcome AUC scores (x):
Outcome: 1: dtd_le_30, auc: 0.867
Outcome: 2: readmissions_in_30_days, auc: 0.626
Outcome AUC scores (t):
Outcome: 1: dtd_le_30, auc: 0.697
Outcome: 2: readmissions_in_30_days, auc: 0.551
Outcome AUC scores (meta):
Outcome: 1: dtd_le_30, auc: 0.873
Outcome: 2: readmissions_in_30_days, auc: 0.619

# 2
Outcome AUC scores (x):
Outcome: 1: dtd_le_30, auc: 0.864
Outcome: 2: readmissions_in_30_days, auc: 0.622
Outcome AUC scores (t):
Outcome: 1: dtd_le_30, auc: 0.707
Outcome: 2: readmissions_in_30_days, auc: 0.598
Outcome AUC scores (meta):
Outcome: 1: dtd_le_30, auc: 0.864
Outcome: 2: readmissions_in_30_days, auc: 0.582

# 3
July 13, 2020 results:
Outcome AUC scores (x):
Outcome: 1: dtd_le_30, auc: 0.866
Outcome: 2: readmissions_in_30_days, auc: 0.592
Outcome AUC scores (t):
Outcome: 1: dtd_le_30, auc: 0.707
Outcome: 2: readmissions_in_30_days, auc: 0.510
Outcome AUC scores (meta):
Outcome: 1: dtd_le_30, auc: 0.868
Outcome: 2: readmissions_in_30_days, auc: 0.602

# 4
July 13, 2020 results:
Outcome AUC scores (x):
Outcome: 1: dtd_le_30, auc: 0.841
Outcome: 2: readmissions_in_30_days, auc: 0.673
Outcome AUC scores (t):
Outcome: 1: dtd_le_30, auc: 0.728
Outcome: 2: readmissions_in_30_days, auc: 0.525
Outcome AUC scores (meta):
Outcome: 1: dtd_le_30, auc: 0.870
Outcome: 2: readmissions_in_30_days, auc: 0.659

# 5
July 13, 2020 results:
Outcome AUC scores (x):
Outcome: 1: dtd_le_30, auc: 0.857
Outcome: 2: readmissions_in_30_days, auc: 0.600
Outcome AUC scores (t):
Outcome: 1: dtd_le_30, auc: 0.673
Outcome: 2: readmissions_in_30_days, auc: 0.549
Outcome AUC scores (meta):
Outcome: 1: dtd_le_30, auc: 0.867
Outcome: 2: readmissions_in_30_days, auc: 0.610

# 6
July 14, 2020 results:
Outcome AUC scores (x):
Outcome: 1: dtd_le_30, auc: 0.830
Outcome: 2: readmissions_in_30_days, auc: 0.629
Outcome AUC scores (t):
Outcome: 1: dtd_le_30, auc: 0.704
Outcome: 2: readmissions_in_30_days, auc: 0.517
Outcome AUC scores (meta):
Outcome: 1: dtd_le_30, auc: 0.841
Outcome: 2: readmissions_in_30_days, auc: 0.628

# 7
July 14, 2020 results:
Outcome AUC scores (x):
Outcome: 1: dtd_le_30, auc: 0.864
Outcome: 2: readmissions_in_30_days, auc: 0.607
Outcome AUC scores (t):
Outcome: 1: dtd_le_30, auc: 0.725
Outcome: 2: readmissions_in_30_days, auc: 0.555
Outcome AUC scores (meta):
Outcome: 1: dtd_le_30, auc: 0.873
Outcome: 2: readmissions_in_30_days, auc: 0.610

# 8
July 14, 2020 results:
Outcome AUC scores (x):
Outcome: 1: dtd_le_30, auc: 0.884
Outcome: 2: readmissions_in_30_days, auc: 0.568
Outcome AUC scores (t):
Outcome: 1: dtd_le_30, auc: 0.730
Outcome: 2: readmissions_in_30_days, auc: 0.564
Outcome AUC scores (meta):
Outcome: 1: dtd_le_30, auc: 0.898
Outcome: 2: readmissions_in_30_days, auc: 0.562

# 9
July 15, 2020 results:
Outcome AUC scores (x):
Outcome: 1: dtd_le_30, auc: 0.875
Outcome: 2: readmissions_in_30_days, auc: 0.639
Outcome AUC scores (t):
Outcome: 1: dtd_le_30, auc: 0.724
Outcome: 2: readmissions_in_30_days, auc: 0.514
Outcome AUC scores (meta):
Outcome: 1: dtd_le_30, auc: 0.888
Outcome: 2: readmissions_in_30_days, auc: 0.649


x_dtd_auc = [.875, .884, .864, .830, .857, .841, .866, .864, .867] Mean: 0.861 std: 0.015
t_dtd_auc = [.724, .730, .725, .704, .673, .728, .707, .707, .697] Mean: 0.710 std: 0.017
meta_dtd_auc = [.888, .898, .873, .841, .867, .870, .868, .864, .873] Mean: 0.871 std: 0.014

x_readmission_auc = [.639, .568, .607, .629, .600, .673, .592, .622, .626] Mean: 0.617 std: 0.028 
t_readmission_auc = [.514, .564, .555, .517, .549, .525, .510, .598, .551] Mean: 0.542 std: 0.027 
meta_readmission_auc = [.649, .562, .610, .628, .610, .659, .602, 0.582, .619] Mean: 0.613 std: 0.028 

'''