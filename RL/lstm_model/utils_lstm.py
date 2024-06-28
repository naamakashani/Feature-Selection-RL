# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 21:29:51 2019

@author: urixs
"""

import numpy as np
import gzip
import struct
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import csv
import pandas as pd
import torch
import scipy.io

def load_tox_171():
    # Load the .mat file
    mat = scipy.io.loadmat(r'C:\Users\kashann\PycharmProjects\choiceMira\data\colon.mat')
    # Display the keys in the .mat file
    X=mat['X']
    y=mat['Y']
    y=y.reshape(-1)
    #change -1 label to 0
    y[y == -1] = 0
    return X, y, y, len(X[0])
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

def load_data_labels():
    # filter_preprocess_X()
    outcomes = pd.read_pickle(r'C:\Users\kashann\PycharmProjects\choiceMira\codeChoice\data_rl\outcomes.pkl')
    n_outcomes = outcomes.shape[1]
    outcome_names = outcomes.columns
    Y = outcomes.to_numpy()
    dtd_indices = [0]  # [i for i, name in enumerate(outcome_names) if 'dtd' in name]
    Y = Y[:, dtd_indices]
    n_outcomes = len(dtd_indices)
    outcome_names = outcome_names[dtd_indices]
    # X_pd = pd.read_pickle(r'C:\Users\kashann\PycharmProjects\choiceMira\codeChoice\data_rl\preprocessed_X.pkl')
    X_pd = pd.read_csv(r'C:\Users\kashann\PycharmProjects\choiceMira\codeChoice\data_rl\preprocessed_X_filtered.csv')
    X = X_pd.to_numpy()
    scaler = StandardScaler()
    # X = scaler.fit_transform(X) #Do not scale if using shap
    Data = pd.read_csv(r'C:\Users\kashann\PycharmProjects\choiceMira\codeChoice\data_rl\new_data_apr22.csv')
    admission_date = pd.to_datetime(Data['Reference Event-Visit Start Date'])

    X = X.astype('float32')
    Y = Y.astype('int')
    Y = Y.reshape(-1)
    return X, Y, X_pd.columns.tolist(), len(X_pd.columns)


def load_data_labels_cut():
    # filter_preprocess_X()
    outcomes = pd.read_pickle(r'C:\Users\kashann\PycharmProjects\choiceMira\codeChoice\data_rl\outcomes.pkl')
    n_outcomes = outcomes.shape[1]
    outcome_names = outcomes.columns
    Y = outcomes.to_numpy()
    dtd_indices = [0]  # [i for i, name in enumerate(outcome_names) if 'dtd' in name]
    Y = Y[:, dtd_indices]
    n_outcomes = len(dtd_indices)
    outcome_names = outcome_names[dtd_indices]
    # X_pd = pd.read_pickle(r'C:\Users\kashann\PycharmProjects\choiceMira\codeChoice\data_rl\preprocessed_X.pkl')
    X_pd = pd.read_csv(r'C:\Users\kashann\PycharmProjects\choiceMira\codeChoice\data_rl\preprocessed_X_filtered.csv')
    X = X_pd.to_numpy()
    scaler = StandardScaler()
    # X = scaler.fit_transform(X) #Do not scale if using shap
    Data = pd.read_csv(r'C:\Users\kashann\PycharmProjects\choiceMira\codeChoice\data_rl\new_data_apr22.csv')
    admission_date = pd.to_datetime(Data['Reference Event-Visit Start Date'])

    X = X.astype('float32')
    Y = Y.astype('int')
    Y = Y.reshape(-1)
    # Finding indices for both label categories
    label_0_indices = np.where(Y == 0)[0]
    label_1_indices = np.where(Y == 1)[0]

    # Count the number of samples in each category
    count_label_0 = len(label_0_indices)
    count_label_1 = len(label_1_indices)

    # Balance the dataset by adjusting samples for each label
    if count_label_0 > count_label_1:
        # Select a random subset of label 0 indices to match label 1 count
        selected_indices = np.random.choice(label_0_indices, count_label_1, replace=False)
        balanced_indices = np.concatenate([selected_indices, label_1_indices])
        np.random.shuffle(balanced_indices)
    else:
        # Select a random subset of label 1 indices to match label 0 count
        selected_indices = np.random.choice(label_1_indices, count_label_0, replace=False)
        balanced_indices = np.concatenate([label_0_indices, selected_indices])
        np.random.shuffle(balanced_indices)

    # Update data and labels with the balanced dataset
    balanced_data = X[balanced_indices]
    balanced_labels = Y[balanced_indices]
    return balanced_data, balanced_labels, X_pd.columns.tolist(), len(X_pd.columns)


def filter_preprocess_X():
    df = pd.read_pickle(r'C:\Users\kashann\PycharmProjects\choiceMira\codeChoice\data_rl\preprocessed_X.pkl')
    # i want to group 3 clumns together by or sign
    df['aspirin'] = df['Medications anticoagulants: ASPIRIN'] | df['Medications anticoagulants: CARTIA'] | df[
        'Medications anticoagulants: MICROPIRIN']
    df.drop(columns=['Medications anticoagulants: ASPIRIN'], inplace=True)
    df.drop(columns=['Medications anticoagulants: CARTIA'], inplace=True)
    df.drop(columns=['Medications anticoagulants: MICROPIRIN'], inplace=True)
    df['Warfarin Sodium'] = df['Medications anticoagulants: COUMADIN'] | df['Medications anticoagulants: HEPARIN']
    df.drop(columns=['Medications anticoagulants: COUMADIN'], inplace=True)
    df.drop(columns=['Medications anticoagulants: HEPARIN'], inplace=True)
    df['clopidogrel'] = df['Medications anticoagulants: CLOPIDOGREL'] | df['Medications anticoagulants: PLAVIX']
    df.drop(columns=['Medications anticoagulants: CLOPIDOGREL'], inplace=True)
    df.drop(columns=['Medications anticoagulants: PLAVIX'], inplace=True)
    df['doxazosin'] = df['Medications hypertnesive: CADEX'] | df['Medications hypertnesive: DOXALOC']
    df.drop(columns=['Medications hypertnesive: CADEX'], inplace=True)
    df.drop(columns=['Medications hypertnesive: DOXALOC'], inplace=True)
    # write df to csv
    df.to_csv(r'C:\Users\kashann\PycharmProjects\choiceMira\codeChoice\data_rl\preprocessed_X_filtered.csv',
              index=False)


def load_data(case):
    if case == 122:  # 50 questions
        data_file = "./Data/small_data50.npy"
        X = np.load(data_file)
        n, d = X.shape
        y = np.load('./Data/labels.npy')
        # standardize features
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X) * 2 - 1
        question_names = np.load('./Data/names_small50.npy')
        class_names = ['no', 'yes']
        print('loaded data,  {} rows, {} columns'.format(n, d))

    if case == 123:  # 100 questions
        data_file = "./Data/small_data100.npy"
        X = np.load(data_file)
        n, d = X.shape
        y = np.load('./Data/labels.npy')
        # standardize features
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X) * 2 - 1
        question_names = np.load('./Data/names_small100.npy')
        class_names = ['no', 'yes']
        print('loaded data,  {} rows, {} columns'.format(n, d))

    return X, y, question_names, class_names, scaler

def load_csv_data():
    data = []
    labels = []
    file_path = "../../ehrData/full_cohort_data.csv"
    # Open the CSV file
    with open(file_path, newline='') as csvfile:
        # Create a CSV reader
        csv_reader = csv.reader(csvfile)
        for line in csv_reader:
            # if it's the first line (header) skip it
            if line[0] == 'age':
                # save the header in npy array for later use
                question_names = np.array(line)
                continue
            # Extract columns from the line
            columns = line
            columns_without_label = columns[0:-1]
            # Replace missing values with the mean of the column
            for i in range(len(columns_without_label)):
                if columns_without_label[i] == '':
                    # Convert other values to float for mean calculation
                    columns_without_label[i] = np.nan
                else:
                    columns_without_label[i] = float(columns_without_label[i])
            # Calculate mean of the column
            column_mean = np.nanmean(columns_without_label)
            # Replace missing values with the mean
            for i in range(len(columns_without_label)):
                if np.isnan(columns_without_label[i]):
                    columns_without_label[i] = column_mean
            data.append(columns_without_label)
            labels.append(int(columns[-1]))

    # Convert lists to NumPy arrays
    X = np.array(data)
    y = np.array(labels)

    n, d = X.shape
    print('Loaded data with {} rows and {} columns'.format(n, d))
    return X, y, question_names, len(columns_without_label)
def load_gisetta():
    data_path = "C:\\Users\\kashann\\PycharmProjects\\choiceMira\\RL\\extra\\gisette\\gisette_train.data"
    labels_path = "C:\\Users\\kashann\\PycharmProjects\\choiceMira\\RL\\extra\\gisette\\gisette_train.labels"
    data = []
    labels = []
    with open(labels_path, newline='') as file:
        # read line by line
        for line in file:
            if int(line) == -1:
                labels.append(0)
            else:
                labels.append(1)
    with open(data_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            sample = []
            columns = row[0].split(' ')
            for i in range(len(columns) - 2):
                sample.append(float(columns[i]))
            data.append(sample)
    X = np.array(data)
    y = np.array(labels)
    return X, y, y, len(sample)


def load_chron():
    data = []
    labels = []
    file_path = '/chron/chron.csv'
    # Open the CSV file
    with open(file_path, newline='') as csvfile:
        # Create a CSV reader
        csv_reader = csv.reader(csvfile)
        for line in csv_reader:
            # if it the first line (header) skip it
            if line[0] == 'Bp':
                # save the header in npy array for later use
                question_names = np.array(line)
                continue
            # columns = [column.split(',') for column in line]
            columns = line
            columns_without_label = columns[0:-1]
            for i in range(len(columns_without_label)):
                columns_without_label[i] = float(columns_without_label[i])
            data.append(columns_without_label)

            labels.append(int(columns[-1]))

    # convet to float each element

    # Convert zero_list to a NumPy array
    X = np.array(data)
    y = np.array(labels)
    n, d = X.shape
    print('loaded data,  {} rows, {} columns'.format(n, d))
    return X, y, question_names, len(columns_without_label)


def load_covid():
    data = []
    labels = []
    file_path = './/extra//covid//covid.csv'
    df = pd.read_csv(file_path)
    df_clean = df.drop(columns=df.columns[(df == 97).any() | (df == 99).any()])
    df_clean['DATE_DIED'] = df_clean['DATE_DIED'].apply(lambda x: 1 if x == '9999-99-99' else 0)
    df_clean_1 = df_clean[df_clean['DATE_DIED'] == 1].sample(frac=0.079)
    df_clean_0 = df_clean[df_clean['DATE_DIED'] == 0]
    df_clean_all = pd.concat([df_clean_0, df_clean_1])
    # change the DATE_DIED column to be the last column in the dataframe
    # save df clean to csv
    file_path_clean = './/extra//covid//covid_clean.csv'
    df_clean_all.to_csv(file_path_clean, index=False)

    # Open the CSV file
    with open(file_path_clean, newline='') as csvfile:
        # Create a CSV reader
        csv_reader = csv.reader(csvfile)
        for line in csv_reader:
            # if it the first line (header) skip it
            if line[0] == 'USMER':
                # save the header in npy array for later use
                question_names = np.array(line)
                continue
            # columns = [column.split(',') for column in line]
            columns = line
            columns_without_label = columns[0:-1]
            for i in range(len(columns_without_label)):
                columns_without_label[i] = float(columns_without_label[i])
            data.append(columns_without_label)
            labels.append(int(columns[-1]))

    X = np.array(data)
    y = np.array(labels)
    n, d = X.shape
    print('loaded data,  {} rows, {} columns'.format(n, d))
    return X, y, question_names, len(columns_without_label)


def create_n_dim():
    # Number of points to generate
    num_points = 2000
    # Generate random x values
    x1_values = np.random.uniform(low=0, high=30, size=num_points)

    # Create y values based on the decision boundary y=-x with some random noise
    x2_values = -x1_values + np.random.normal(0, 2, size=num_points)

    # Create labels based on the side of the decision boundary
    labels = np.where(x2_values > -1 * x1_values, 1, 0)
    # create numpy of zeros
    X = np.zeros((num_points, 10))
    i = 0
    while i < num_points:
        # choose random index to assign x1 and x2 values
        index = np.random.randint(0, 10)
        # assign x1 to index for 5 samples
        X[i][index] = x1_values[i]
        X[i + 1][index] = x1_values[i + 1]
        X[i + 2][index] = x1_values[i + 2]
        X[i + 3][index] = x1_values[i + 3]
        X[i + 4][index] = x1_values[i + 4]
        # choose random index to assign x2 that is not the same as x1
        index2 = np.random.randint(0, 10)
        while index2 == index:
            index2 = np.random.randint(0, 10)
        X[i][index2] = x2_values[i]
        X[i + 1][index2] = x2_values[i + 1]
        X[i + 2][index2] = x2_values[i + 2]
        X[i + 3][index2] = x2_values[i + 3]
        X[i + 4][index2] = x2_values[i + 4]
        i += 5
    question_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    return X, labels, question_names, 10


def load_diabetes():
    data = []
    labels = []
    file_path = 'C:\\Users\\kashann\\PycharmProjects\\choiceMira\\RL\\extra\\diabetes\\diabetes_prediction_dataset.csv'
    df = pd.read_csv(file_path)
    df_1 = df[df['diabetes'] == 1]
    df_0 = df[df['diabetes'] == 0].sample(frac=0.092)
    df_all = pd.concat([df_0, df_1])

    # save df clean to csv
    file_path_clean = 'C:\\Users\\kashann\\PycharmProjects\\choiceMira\\RL\\extra\\diabetes\\diabetes_clean.csv'
    df_all.to_csv(file_path_clean, index=False)
    # Open the CSV file
    with open(file_path_clean, newline='') as csvfile:
        # Create a CSV reader
        csv_reader = csv.reader(csvfile)
        for line in csv_reader:
            # if it the first line (header) skip it
            if line[0] == 'gender':
                # save the header in npy array for later use
                question_names = np.array(line)
                continue
            # columns = [column.split(',') for column in line]
            columns = line
            columns_without_label = columns[0:-1]
            if columns_without_label[0] == "Female":
                columns_without_label[0] = 0
            else:
                columns_without_label[0] = 1
            if columns_without_label[4] == "never":
                columns_without_label[4] = 0
            if columns_without_label[4] == "former":
                columns_without_label[4] = 1
            if columns_without_label[4] == "current":
                columns_without_label[4] = 2
            if columns_without_label[4] == "No Info":
                columns_without_label[4] = 3
            if columns_without_label[4] == "not current":
                columns_without_label[4] = 4
            if columns_without_label[4] == "ever":
                columns_without_label[4] = 5
            for i in range(len(columns_without_label)):
                columns_without_label[i] = float(columns_without_label[i])

            data.append(columns_without_label)

            labels.append(int(columns[-1]))

    # Convert zero_list to a NumPy array
    X = np.array(data)
    y = np.array(labels)
    n, d = X.shape
    # standardize features
    # scaler = MinMaxScaler()
    # X = scaler.fit_transform(X) * 2 - 1
    class_names = [0, 1]
    print('loaded data,  {} rows, {} columns'.format(n, d))
    return X, y, question_names, len(columns_without_label)


def diabetes_prob_actions():
    cost_list = np.array(np.ones(9))
    return torch.from_numpy(np.array(cost_list))


def prob_rec():
    cost_list = np.array(np.ones(3))
    return torch.from_numpy(np.array(cost_list))


def prob_actions():
    cost_list = np.array(np.ones(32))
    return torch.from_numpy(np.array(cost_list))


def covid_prob_actions():
    cost_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    return torch.from_numpy(np.array(cost_list))


def load_mnist(case=1):
    if os.path.exists('./mnist_check/X_test.npy'):
        X_test = np.load('./mnist_check/X_test.npy')
    else:
        X_test = read_idx('./mnist/t10k-images-idx3-ubyte.gz')
        X_test = X_test.reshape(-1, 28 * 28)
        np.save('./mnist/X_test.npy', X_test)
    if os.path.exists('./mnist_check/X_train.npy'):
        X_train = np.load('./mnist_check/X_train.npy')
    else:
        X_train = read_idx('./mnist/train-images-idx3-ubyte.gz')
        X_train = X_train.reshape(-1, 28 * 28)
        np.save('./mnist/X_train.npy', X_train)
    if os.path.exists('./mnist_check/y_test.npy'):
        y_test = np.load('./mnist_check/y_test.npy')
    else:
        y_test = read_idx('./mnist/t10k-labels-idx1-ubyte.gz')
        np.save('./mnist/y_test.npy', y_test)
    if os.path.exists('./mnist/y_train.npy'):
        y_train = np.load('./mnist/y_train.npy')
    else:
        y_train = read_idx('./mnist_check/train-labels-idx1-ubyte.gz')
        np.save('./mnist_check/y_train.npy', y_train)

    if case == 1:  # small version
        train_inds = y_train <= 2
        test_inds = y_test <= 2
        X_train = X_train[train_inds]
        X_test = X_test[test_inds]
        y_train = y_train[train_inds]
        y_test = y_test[test_inds]

    return X_train / 127.5 - 1., X_test / 127.5 - 1, y_train, y_test


def load_mi_scores():
    '''
    if os.path.exists('./mnist/mi.npy'):
        print('Loading stored mutual information scores')
        return np.load('./mnist/mi.npy')
    else:
        return None
    '''
    X_train, X_test, y_train, y_test = load_mnist(case=2)
    max_depth = 5

    # define a decision tree classifier
    clf = DecisionTreeClassifier(max_depth=max_depth)

    # fit model
    clf = clf.fit(X_train, y_train)
    return clf.feature_importances_


def read_idx(filename):
    with gzip.open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def plot_mnist_digit(digit,
                     guess,
                     true_label,
                     num_steps,
                     save=True,
                     fig_num=0,
                     save_dir='.',
                     actions=None):
    import matplotlib.pyplot as plt
    digit = digit.reshape(28, 28)
    fig, ax = plt.subplots()
    ax.set_title('true label: {}, guess: {}, num steps: {}'.format(true_label, guess, num_steps), fontsize=18)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    im = ax.imshow(digit, cmap='gray')
    if actions is not None:
        for i, a in enumerate(actions):
            if a != 784:
                row = a % 28
                col = int(a / 28)
                text = ax.text(row, col - 2, i + 1, ha="center", va="center", color="b", size=15)
    plt.show()
    if save:
        fig.savefig(save_dir + '/im_' + str(fig_num) + '.png')


def plot_medical(digit,
                 guess,
                 true_label,
                 num_steps,
                 save=True,
                 fig_num=0,
                 save_dir='.',
                 actions=None):
    import matplotlib.pyplot as plt
    digit = digit.reshape(64, 64)
    fig, ax = plt.subplots()
    ax.set_title('true label: {}, guess: {}, num steps: {}'.format(true_label, guess, num_steps), fontsize=18)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    im = ax.imshow(digit, cmap='gray')
    if actions is not None:
        for i, a in enumerate(actions):
            if a != 64 * 64:
                row = a % 64
                col = int(a / 64)
                text = ax.text(row, col - 2, i + 1, ha="center", va="center", color="b", size=15)
    plt.show()
    if save:
        fig.savefig(save_dir + '/im_' + str(fig_num) + '.png')


def scale_individual_value(val, ind, scaler):
    return (val - scaler.data_min_[ind]) / (scaler.data_max_[ind] - scaler.data_min_[ind]) * 2. - 1.
