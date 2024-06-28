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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy DT: {accuracy}")
    print(clf.get_depth())

    selected_paths = get_selected_features(clf.tree_, X_test)

    # Calculate intersection and union of selected paths
    intersection = set.intersection(*selected_paths)
    union = set.union(*selected_paths)
    return accuracy, clf.get_depth(), len(intersection),len(union)

def check_DT():
    total_accuracy = 0
    total_depth = 0
    total_intersection =0
    total_union = 0
    list_accuracy = []
    total_depth_list = []
    total_intersection_lies=[]
    total_union_lies = []
    for i in range(10):
        #calc the mean of the accuracy and depth
        accuracy, depth, intersection, union = decision_tree()
        list_accuracy.append(accuracy)
        total_depth_list.append(depth)
        total_intersection_lies.append(intersection)
        total_union_lies.append(union)
        total_accuracy += accuracy
        total_depth += depth
        total_intersection += intersection
        total_union += union
    print("Average accuracy: ", total_accuracy/10)
    print("Average depth: ", total_depth/10)
    print("Average intersection: ", total_intersection/10)
    print("Average union: ", total_union/10)
    #print standard dev of list_accuracy
    print("Standard deviation of accuracy: ", np.std(list_accuracy))
    print ("Standard deviation of depth: ", np.std(total_depth_list))
    print("Standard deviation of intersection: ", np.std(total_intersection_lies))
    print("Standard deviation of union: ", np.std(total_union_lies))











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
    X, y, _, number_of_features = utils.load_csv_data()
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
    print( "XGBOOST ACCURACY")
    print(accuracy)


def main():
    # XGboost_new()
    check_DT()






if __name__ == "__main__":
    os.chdir("C:\\Users\\kashann\\PycharmProjects\\RLadaptive\\RL\\")
    main()


