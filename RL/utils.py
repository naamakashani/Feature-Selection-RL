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
from sklearn.utils import resample
from ucimlrepo import fetch_ucirepo
import scipy.io
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


def load_mnist_data():
    # read npy file
    train_x = "C:\\Users\\kashann\\PycharmProjects\\choiceMira\\RL\\extra\\mnist_check\\X_train.npy"
    X_train = np.load(train_x)
    test_x = "C:\\Users\\kashann\\PycharmProjects\\choiceMira\\RL\\extra\\mnist_check\\X_test.npy"
    X_test = np.load(test_x)
    y_train = "C:\\Users\\kashann\\PycharmProjects\\choiceMira\\RL\\extra\\mnist_check\\y_train.npy"
    y_train = np.load(y_train)
    y_test = "C:\\Users\\kashann\\PycharmProjects\\choiceMira\\RL\\extra\\mnist_check\\y_test.npy"
    y_test = np.load(y_test)
    # combine X_train and X_test
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    return X, y, y, len(X[0])


def load_FS_data():
    # filter_preprocess_X()
    outcomes = pd.read_pickle(r'C:\Users\kashann\PycharmProjects\choiceMira\ehrData\diabetes_all.pkl')
    X = outcomes['features']
    y = outcomes['targets']
    cost = outcomes['costs']
    return X, y, cost, len(X[0])

def load_ucimlrepo():
    # fetch dataset
    diabetic_retinopathy_debrecen = fetch_ucirepo(id=329)
    X = diabetic_retinopathy_debrecen.data.features.to_numpy()
    y = diabetic_retinopathy_debrecen.data.targets.to_numpy()
    y = y.squeeze()
    y = y.tolist()
    y = np.array(y)
    return X, y, diabetic_retinopathy_debrecen.metadata.num_features, diabetic_retinopathy_debrecen.metadata.num_features


def load_fetal():
    file_path= r'C:\Users\kashann\PycharmProjects\choiceMira\data\fetal_health_None.csv'
    df = pd.read_csv(file_path)
    Y = df['fetal_health'].to_numpy().reshape(-1)
    X = df.drop(df.columns[-1], axis=1).to_numpy()
    return X,Y,Y,len(X[0])

def load_madelon():
    madelon_train = r'C:\Users\kashann\PycharmProjects\choiceMira\data\MADELON\madelon_train.data'
    madelon_train_labels = r'C:\Users\kashann\PycharmProjects\choiceMira\data\MADELON\madelon_train.labels'
    madelon_valid = r'C:\Users\kashann\PycharmProjects\choiceMira\data\MADELON\madelon_valid.data'
    madelon_valid_labels = r'C:\Users\kashann\PycharmProjects\choiceMira\data\MADELON\madelon_valid.labels'
    madelon_train_df = pd.read_csv(madelon_train, delimiter=' ', header=None)
    madelon_train_labels_df = pd.read_csv(madelon_train_labels, delimiter=' ', header=None)
    madelon_valid_df = pd.read_csv(madelon_valid, delimiter=' ', header=None)
    madelon_valid_labels_df = pd.read_csv(madelon_valid_labels, delimiter=' ', header=None)
    #concat train and valid
    X = pd.concat([madelon_train_df, madelon_valid_df])
    y = pd.concat([madelon_train_labels_df, madelon_valid_labels_df])
    for i in range(1, len(X)):
        for j in range(len(X.columns)):
            if np.random.rand() < 0.2:
                X.iloc[i, j] = np.nan
    #remove the last colum in X
    X = X.iloc[:, :-1]
    X=X.to_numpy()
    y=y.to_numpy()
    y=y.reshape(-1)
    #change -1 label to 0
    y[y == -1] = 0
    return X, y, y, len(X[0])


import scipy.io
import numpy as np


def load_colon():
    # Load the .mat file
    mat = scipy.io.loadmat(r'C:\Users\kashann\PycharmProjects\choiceMira\data\colon.mat')
    # Extract features X and labels y
    X = mat['X']
    y = mat['Y']
    y = y.reshape(-1)
    # Change -1 label to 0
    y[y == -1] = 0
    df = pd.DataFrame(X)
    for i in range(0, len(df)):
        for j in range(len(df.columns)-1):
            if np.random.rand() < 0.8:
                df.iloc[i, j] = np.nan

    X = df.to_numpy()
    return X, y, y, len(X[0])
    # File path for the new data with None values



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
    return X, Y, X_pd.columns, len(X_pd.columns)


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


def load_heart():
    data = []
    labels = []
    file_path = "/RL/extra/heart.csv"
    # Open the CSV file
    with open(file_path, newline='') as csvfile:
        # Create a CSV reader
        csv_reader = csv.reader(csvfile)
        for line in csv_reader:
            # if it the first line (header) skip it
            if line[0] == 'age':
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
    class_names = [0, 1]
    print('loaded data,  {} rows, {} columns'.format(n, d))
    return X, y, question_names, len(columns_without_label)


def load_dermatology():
    data = []
    labels = []
    file_path = 'C:\\Users\\kashann\\PycharmProjects\\choiceMira\\RL\\extra\\dermatology_database.csv'
    # Open the CSV file
    with open(file_path, newline='') as csvfile:
        # Create a CSV reader
        csv_reader = csv.reader(csvfile)
        for line in csv_reader:
            if line[0] == 'erythema':
                question_names = np.array(line)
                continue

            columns = line
            columns_without_label = columns[0:-1]
            for i in range(len(columns_without_label)):
                if columns_without_label[i] == '?':
                    columns_without_label[i] = 0
                columns_without_label[i] = float(columns_without_label[i])
            data.append(columns_without_label)
            labels.append(int(float((columns[-1]))))

    # convet to float each element
    # balance data and labels in multiclass classification

    # Convert zero_list to a NumPy array
    X = np.array(data)
    y = np.array(labels)

    n, d = X.shape
    class_names = [0, 1]
    print('loaded data,  {} rows, {} columns'.format(n, d))
    return X, y, question_names, len(columns_without_label)


def load_ehr():
    labels = []
    data = []
    file_path = "C:\\Users\\kashann\\PycharmProjects\\choiceMira\\ehrData\\data-ori.csv"
    # Open the CSV file
    with open(file_path, newline='') as csvfile:
        # Create a CSV reader
        csv_reader = csv.reader(csvfile)
        for line in csv_reader:
            if line[0] == 'HAEMATOCRIT':
                question_names = np.array(line)
                continue
            columns = line
            columns_without_label = columns[0:-1]
            for i in range(len(columns_without_label)):
                if columns_without_label[i] == 'M':
                    columns_without_label[i] = 0
                if columns_without_label[i] == 'F':
                    columns_without_label[i] = 1
                columns_without_label[i] = float(columns_without_label[i])
            data.append(columns_without_label)
            if columns[-1] == "out":
                label = 0
            else:
                label = 1
            labels.append(float(label))

    # convet to float each element
    # balance data and labels in multiclass classification

    # Convert zero_list to a NumPy array
    X = np.array(data)
    y = np.array(labels)

    n, d = X.shape
    class_names = [0, 1]
    print('loaded data,  {} rows, {} columns'.format(n, d))
    return X, y, question_names, len(columns_without_label)


def load_sonar():
    labels = []
    data = []
    file_path = "C:\\Users\\kashann\\PycharmProjects\\choiceMira\\ehrData\\sonar.all-data.csv"
    # Open the CSV file
    with open(file_path, newline='') as csvfile:
        # Create a CSV reader
        csv_reader = csv.reader(csvfile)
        for line in csv_reader:
            if line[0] == 'Freq_1':
                question_names = np.array(line)
                continue
            columns = line
            columns_without_label = columns[0:-1]
            for i in range(len(columns_without_label)):
                if columns_without_label[i] == 'yes':
                    columns_without_label[i] = 1
                if columns_without_label[i] == 'no':
                    columns_without_label[i] = 0
                if columns_without_label[i] == 'female':
                    columns_without_label[i] = 0
                if columns_without_label[i] == 'male':
                    columns_without_label[i] = 1

            data.append(columns_without_label)
            if columns[-1] == "R":
                label = 0
            else:
                label = 1
            labels.append(float(label))

    # convet to float each element
    # balance data and labels in multiclass classification

    # Convert zero_list to a NumPy array
    X = np.array(data)
    y = np.array(labels)

    n, d = X.shape
    class_names = [0, 1]
    print('loaded data,  {} rows, {} columns'.format(n, d))
    return X, y, question_names, len(columns_without_label)

def load_alzhimer():
    df = pd.read_csv("C:\\Users\\kashann\\PycharmProjects\\choiceMira\\data\\alzhimer.csv")
    df_dropped = df.drop(df.columns[0], axis=1)
    df_dropped.iloc[:, -1] = df_dropped.iloc[:, -1].map({'P': 1, 'H': 0})
    Y = df_dropped.iloc[:, -1]
    df_dropped = df_dropped.drop(df_dropped.columns[-1], axis=1)
    df_normalized = df_dropped.apply(lambda x: 2 * ((x - x.min()) / (x.max() - x.min())) - 1)

    X = df_normalized.astype(float)
    X = X.to_numpy()
    Y = Y.to_numpy()
    return X, Y, df_normalized.columns, len(df_normalized.columns)


def load_student():
    df = pd.read_csv("C:\\Users\\kashann\\PycharmProjects\\choiceMira\\ehrData\\student.csv")

    # Drop rows with missing values
    df = df.dropna()

    mapping = {
        'Age (4 levels)': {'less 18': 0, '18': 1, '19': 2, '20 and more': 3},
        'Gender': {'female': 0, 'male': 1},
        'French nationality': {'no': 0, 'yes': 1},
        'Field of study': {'other programs': 0, 'medicine and allied programs': 1, 'humanities': 2,
                           'sciences': 3, 'sports science': 4, 'law and political sciences': 5},
        'Year of university': {'first': 1, 'second': 2, 'third': 3},
        'Learning disabilities': {'no': 0, 'yes': 1},
        'Difficulty memorizing lessons': {'no': 0, 'yes': 1},
        'Professional objective': {'no': 0, 'yes': 1},
        'Informed about opportunities': {'no': 0, 'yes': 1},
        'Satisfied with living conditions': {'no': 0, 'yes': 1},
        'Living with a partner/child': {'no': 0, 'yes': 1},
        'Parental home': {'no': 0, 'yes': 1},
        'Having only one parent': {'no': 0, 'yes': 1},
        'At least one parent unemployed': {'no': 0, 'yes': 1},
        'Siblings': {'no': 0, 'yes': 1},
        'Long commute': {'no': 0, 'yes': 1},
        'Mode of transportation': {'by public transportation': 0, 'on foot': 1, 'by car': 2},
        'Financial difficulties': {'no': 0, 'yes': 1},
        'Grant': {'no': 0, 'yes': 1},
        'Additional income': {'no': 0, 'yes': 1},
        'Public health insurance ': {'no': 0, 'yes': 1},
        'Private health insurance ': {'no': 0, 'yes': 1},
        'C.M.U.': {'no': 0, 'yes': 1},
        'Irregular rhythm of meals': {'no': 0, 'yes': 1},
        'Unbalanced meals': {'no': 0, 'yes': 1},
        'Eating junk food': {'no': 0, 'yes': 1},
        'On a diet': {'no': 0, 'yes': 1},
        'Irregular rhythm or unbalanced meals': {'no': 0, 'yes': 1},
        'Physical activity(3 levels)': {'no': 0, 'occasionally': 1, 'regularly': 2},
        'Physical activity(2 levels)': {'no activity or occasionally': 0, 'regularly': 1},
        'Overweight and obesity': {'no': 0, 'yes': 1},
        'Prehypertension or hypertension': {'no': 0, 'yes': 1},
        'Abnormal heart rate': {'no': 0, 'yes': 1},
        'Decreased in distant visual acuity': {'no': 0, 'yes': 1},
        'Decreased in close visual acuity': {'no': 0, 'yes': 1},
        'Urinalysis (hematuria)': {'no': 0, 'yes': 1},
        'Urinalysis leukocyturia)': {'no': 0, 'yes': 1},
        'Urinalysis (glycosuria)': {'no': 0},
        'Urinalysis (proteinuria)': {'no': 0},
        'Urinalysis (positive nitrite test)': {'no': 0},
        'Abnormal urinalysis': {'no': 0, 'yes': 1},
        'Vaccination up to date': {'no': 0, 'yes': 1},
        'Control examination needed': {'no': 0, 'yes': 1},
        'Anxiety symptoms': {'no': 0, 'yes': 1},
        'Panic attack symptoms': {'no': 0, 'yes': 1},
        'Depressive symptoms': {'no': 0, 'yes': 1},
        'Cigarette smoker (5 levels)': {'no': 0, 'regularly': 1, 'occasionally': 2, 'frequently': 3, 'heavily': 4},
        'Cigarette smoker (3 levels)': {'no': 0, 'occasionally to regularly': 1, 'frequently to heavily': 2},
        'Drinker (3 levels)': {'no': 0, 'occasionally': 1, 'regularly': 2},
        'Drinker (2 levels)': {'no or occasionally': 0, 'regularly to heavily': 1},
        'Binge drinking': {'no': 0, 'yes': 1},
        'Marijuana use': {'no': 0, 'yes': 1},
        'Other recreational drugs': {'no': 0, 'yes': 1}
    }

    # Apply mapping to non-numeric columns
    for column, map_dict in mapping.items():
        df[column] = df[column].map(map_dict)
    df = df.astype(float)
    X = df.drop(columns=['Depressive symptoms'])
    Y = df['Depressive symptoms']
    X = X.to_numpy()
    Y = Y.to_numpy()
    return X, Y, df.columns, len(df.columns)-1


def import_breast():
    from ucimlrepo import fetch_ucirepo

    # fetch dataset
    breast_cancer_wisconsin_prognostic = fetch_ucirepo(id=16)

    # data (as pandas dataframes)
    X = breast_cancer_wisconsin_prognostic.data.features.to_numpy()
    y = breast_cancer_wisconsin_prognostic.data.targets.to_numpy()
    y[y == 'R'] = 1
    y[y == 'N'] = 0
    X[X == 'nan'] = 0
    X = np.nan_to_num(X, nan=0)
    y = y.squeeze()
    y = y.tolist()
    y = np.array(y)

    return X, y, breast_cancer_wisconsin_prognostic.metadata.num_features, breast_cancer_wisconsin_prognostic.metadata.num_features


def create_data():
    # Number of points to generate
    num_points = 100
    # Generate random x values
    x1_values = np.random.uniform(low=-30, high=30, size=num_points)
    x2_values = np.random.uniform(low=-30, high=30, size=num_points)
    x3_values = np.random.uniform(low=-30, high=30, size=num_points)
    # Create y values based on the decision boundary if x+y > 9 then 1 else 0
    labels = np.where((x1_values + x2_values + x3_values > 10), 1, 0)
    # Split the data into training and testing sets
    x = np.column_stack((x1_values, x2_values, x3_values))
    return x, labels, x1_values, 3





def create():
    # Set a random seed for reproducibility

    # Number of points to generate
    num_points = 100

    # Generate random x values
    x1_values = np.random.uniform(low=0, high=30, size=num_points)

    # Create y values based on the decision boundary y=-x with some random noise
    x2_values = -x1_values + np.random.normal(0, 2, size=num_points)

    # Create labels based on the side of the decision boundary
    labels = np.where(x2_values > -1 * x1_values, 1, 0)

    # Create a scatter plot of the dataset with color-coded labels
    plt.scatter(x1_values, x2_values, c=labels, cmap='viridis', marker='o', label='Data Points')
    # Split the data into training and testing sets
    x = np.column_stack((x1_values, x2_values))
    return x, labels, x1_values, 3


def balance_class_multi(X, y, noise_std=0.01):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    max_class_count = np.max(class_counts)

    # Calculate the difference in sample counts for each class
    count_diff = max_class_count - class_counts

    # Initialize arrays to store balanced data
    X_balanced = X.copy()
    y_balanced = y.copy()

    # Add noise to the features of the minority classes to balance the dataset
    for minority_class, diff in zip(unique_classes, count_diff):
        if diff > 0:
            # Get indices of samples belonging to the current minority class
            minority_indices = np.where(y == minority_class)[0]

            # Randomly sample indices from the minority class to add noise
            noisy_indices = np.random.choice(minority_indices, diff, replace=True)

            # Add noise to the features of the selected samples
            X_balanced = np.concatenate([X_balanced, add_noise(X[noisy_indices], noise_std)], axis=0)
            y_balanced = np.concatenate([y_balanced, y[noisy_indices]], axis=0)

    return X_balanced, y_balanced


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


def load_csv_data():
    data = []
    labels = []
    file_path = "C:\\Users\\kashann\\PycharmProjects\\choiceMira\\ehrData\\full_cohort_data.csv"
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


def load_diabetes():
    file_path = r'C:\Users\kashann\PycharmProjects\choiceMira\data\diabetes_None.csv'
    data = []
    labels = []
    df = pd.read_csv(file_path)
    # df_1 = df[df['diabetes'] == 1]
    # df_0 = df[df['diabetes'] == 0].sample(frac=0.092)
    # df_all = pd.concat([df_0, df_1])

    # save df clean to csv
    file_path_clean = 'C:\\Users\\kashann\\PycharmProjects\\choiceMira\\RL\\extra\\diabetes\\diabetes_clean.csv'
    df.to_csv(file_path_clean, index=False)
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
                if columns_without_label[i] != '':
                    columns_without_label[i] = float(columns_without_label[i])
                else:
                    columns_without_label[i] = 0

            data.append(columns_without_label)

            labels.append(int(float(columns[-1])))

    # Convert zero_list to a NumPy array
    X = np.array(data)
    y = np.array(labels)

    return X, y, question_names, len(columns_without_label)


def create_demo():
    np.random.seed(34)
    Xs1 = np.random.normal(loc=1,scale=0.5,size=(300,5))
    Ys1 = -2*Xs1[:,0]+1*Xs1[:,1]-0.5*Xs1[:,2]
    Xs2 = np.random.normal(loc=-1,scale=0.5,size=(300,5))
    Ys2 = -0.5*Xs2[:,2]+1*Xs2[:,3]-2*Xs2[:,4]
    X_data = np.concatenate((Xs1, Xs2), axis=0)
    Y_data = np.concatenate((Ys1.reshape(-1, 1), Ys2.reshape(-1, 1)), axis=0)
    Y_data = Y_data-Y_data.min()
    Y_data=Y_data/Y_data.max()
    case_labels = np.concatenate((np.array([1] * 300), np.array([2] * 300)))
    Y_data = np.concatenate((Y_data, case_labels.reshape(-1, 1)), axis=1)
    Y_data = Y_data[:, 0].reshape(-1, 1)
    return X_data,Y_data,5,5

def diabetes_prob_actions():
    cost_list = np.array(np.ones(9))
    return torch.from_numpy(np.array(cost_list))
def load_cost():
    #create numpy array of costs with initalize values

    cost_list = np.array([1,1,3,1,1,2,3,3])
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


def process_images_to_npy():
    if os.path.isfile('./processed_data.npy'):
        # load data from file
        data = np.load('./processed_data.npy')
        if os.path.isfile('processed_labels.npy'):
            # load labels from file
            labels = np.load('processed_labels.npy')
            for i in range(len(labels)):
                if labels[i] == "AbdomenCT":
                    labels[i] = 0
                elif labels[i] == "Hand":
                    labels[i] = 1
                else:
                    labels[i] = 2
    else:

        files_path = []
        files_labels = []
        PATH = 'input/medical-mnist'

        for root, dirs, files in os.walk(PATH):
            p = pathlib.Path(root)
            for file in files:
                files_path.append(root + '/' + file)
                files_labels.append(p.parts[-1])

        data = []

        for path in files_path:
            img = Image.open(path)
            img.load()
            img_X = np.asarray(img, dtype=np.int16)
            data.append(img_X)

        data = np.array(data)
        labels = np.array(files_labels)
        for i in range(len(labels)):
            if labels[i] == "AbdomenCT":
                labels[i] = 0
            elif labels[i] == "Hand":
                labels[i] = 1
            else:
                labels[i] = 2

        # Save data and labels to .npy files
        np.save('processed_data.npy', data)
        np.save('processed_labels.npy', labels)

    return data, labels


def load_medical_scores():
    '''
    if os.path.exists('./mnist/mi.npy'):
        print('Loading stored mutual information scores')
        return np.load('./mnist/mi.npy')
    else:
        return None
    '''
    data, labels = process_images_to_npy()
    data = data.reshape(-1, 64 * 64)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.33)
    max_depth = 5

    # define a decision tree classifier
    clf = DecisionTreeClassifier(max_depth=max_depth)

    # fit model
    clf = clf.fit(X_train, y_train)
    return clf.feature_importances_


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
