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
def get_selected_features(tree, X):
    # Traverse the tree for each sample in X and collect selected features
    selected_features = []
    for sample in X:
        node = 0
        features = set()
        while tree.children_left[node] != -1:
            feature = tree.feature[node]
            features.add(feature)
            threshold = tree.threshold[node]
            if sample[feature] <= threshold:
                node = tree.children_left[node]
            else:
                node = tree.children_right[node]
        selected_features.append(features)
    return selected_features

def decision_tree():
    # Splitting data into training and testing sets
    X, y, _, number_of_features = utils.load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = DecisionTreeClassifier(max_depth=2)  # You can specify hyperparameters here
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(clf.get_depth())
    print(f"Number of features: {number_of_features}")
    selected_paths = get_selected_features(clf.tree_, X_test)

    # Calculate intersection and union of selected paths
    intersection = set.intersection(*selected_paths)
    union = set.union(*selected_paths)

    print("Intersection of selected paths:", intersection)
    print("Union of selected paths:", union)




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
    X_train, X_test, y_train, y_test = utils.load_diabetes()
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


import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import numpy as np


def get_selected_features_xgboost(model, threshold):
    # Get feature importance from the trained XGBoost model
    importance = model.get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    # Extract selected features based on threshold
    selected_features = [feature for feature, importance in sorted_importance if importance >= threshold]
    return selected_features


def XGboost_new():
    X, y, _, number_of_features = utils.load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
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

    # Get selected features based on feature importance
    selected_features = get_selected_features_xgboost(model, threshold)
    return accuracy, selected_features





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
    decision_tree()


    # Calculate intersection and union with previously calculated selected paths
    # Call the XGboost function
    accuracy_xgboost, selected_features_xgboost = XGboost_new()

    # Calculate intersection and union with previously calculated selected paths (if available)
    intersection = set(selected_features_xgboost).intersection(*selected_features_xgboost)
    union = set(selected_features_xgboost).union(*selected_features_xgboost)

    print("Intersection of selected features from XGBoost:", intersection)
    print("Union of selected features from XGBoost:", union)

    print("Intersection of selected features from XGBoost and Decision Tree:", intersection)
    print("Union of selected features from XGBoost and Decision Tree:", union)
