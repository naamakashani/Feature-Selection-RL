from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier  # For classification tasks
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # For classification tasks
import RL.utils as utils
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris  # Example dataset (you can use your own dataset)
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


def decision_tree():
    # Splitting data into training and testing sets
    X, y, _, number_of_features = utils.load_data_labels()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = DecisionTreeClassifier(max_depth=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Number of features: {number_of_features}")


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
    X_train, X_test, y_train, y_test = train_test_split(
        np.column_stack((x1_values, x2_values, x3_values)), labels, test_size=0.2,
        random_state=42)
    return X_train, X_test, y_train, y_test


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
    # show the plot
    # plt.show()
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(np.column_stack((x1_values, x2_values)), labels, test_size=0.2,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


def XGboost():
    X_train, X_test, y_train, y_test = create()
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {
        'objective': 'binary:logistic',  # for binary classification
        'max_depth': 2,
        'learning_rate': 0.1,
        'eval_metric': 'logloss'
    }
    num_rounds = 50  # Number of boosting rounds
    model = xgb.train(params, dtrain, num_rounds)
    y_pred = model.predict(dtest)
    threshold = 0.5  # Adjust threshold for binary classification
    y_pred_binary = [1 if pred > threshold else 0 for pred in y_pred]

    accuracy = accuracy_score(y_test, y_pred_binary)
    return accuracy


def cut_decition_tree_():
    X, y, _, number_of_features = utils.load_data_labels()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = DecisionTreeClassifier(max_depth=1)  # You can specify hyperparameters here
    clf.fit(X_train, y_train)
    # check depth
    print(clf.get_depth())
    # print the feature that chosen
    print(clf.tree_.feature)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")


def cut_decision_tree():
    # Split the data into training and testing sets (80% train, 20% test)
    X, y, _, number_of_features = utils.load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    # Create a decision tree classifier with a maximum depth of 2 features
    tree_classifier = DecisionTreeClassifier(max_depth=2)

    # Train the decision tree classifier on the training data
    tree_classifier.fit(X_train, y_train)
    # print the depth of the tree

    # Predict using the trained model on the test set
    y_pred = tree_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, tree_classifier.get_depth()


def knn():
    X, y, _, number_of_features = utils.load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)  # You can change the number of neighbors (K) as needed
    # Train the KNN classifier on the training data
    knn.fit(X_train, y_train)
    # Make predictions on the test data
    y_pred = knn.predict(X_test)
    # Calculate accuracy manually
    accuracy_manual = accuracy_score(y_test, y_pred)
    print(f"Accuracy of KNN: {accuracy_manual:.3f}")


def SVM():
    # Initialize the SVM classifier
    X, y, _, number_of_features = utils.load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    svm = SVC(kernel='rbf', C=1.0, gamma='scale')  # You can adjust kernel, C, gamma as needed
    # Train the SVM classifier on the training data
    svm.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred_svm = svm.predict(X_test)

    # Calculate accuracy using accuracy_score
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print(f"Accuracy of SVM: {accuracy_svm:.3f}")


def main():
    sum = 0
    sum_depth = 0
    for i in range(10):
        acc, depth = cut_decision_tree()
        sum += acc
        sum_depth += depth
    print(sum / 10)
    print (sum_depth / 10)


if __name__ == "__main__":
    os.chdir("C:\\Users\\kashann\\PycharmProjects\\RLadaptive\\RL\\")
    main()
