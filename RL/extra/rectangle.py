import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree


def decision_tree(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = DecisionTreeClassifier(random_state=42)  # You can specify hyperparameters here
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")


# Function to generate a synthetic dataset
def generate_dataset(num_samples):
    np.random.seed(42)

    # Generate features
    features = np.random.rand(num_samples, 2) * 2 - 1  # Random points in the [-1, 1] range

    # Generate labels based on the slant of a rectangle
    labels = np.zeros(num_samples, dtype=int)
    for i in range(num_samples):
        x, y = features[i]
        if x > y:
            labels[i] = 1

    return features, labels


np.random.seed(42)


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
    X_train, X_test, y_train, y_test = train_test_split(np.column_stack((x1_values, x2_values)), labels, test_size=0.2,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


def create_n_dim():
        # Number of points to generate
        num_points = 2000

        # Generate random x values
        x1_values = np.random.uniform(low=0, high=30, size=num_points)

        # Create y values based on the decision boundary y=-x with some random noise
        x2_values = -x1_values + np.random.normal(0, 2, size=num_points)

        # Create labels based on the side of the decision boundary
        labels = np.where(x2_values > -1 * x1_values, 1, 0)
        #create numpy of zeros
        X = np.zeros((num_points,10 ))
        i=0
        while i < num_points:
            #choose random index to assign x1 and x2 values
            index = np.random.randint(0, 10)
            #assign x1 to index for 5 samples
            X[i][index] = x1_values[i]
            X[i+1][index] = x1_values[i+1]
            X[i+2][index] = x1_values[i+2]
            X[i+3][index] = x1_values[i+3]
            X[i+4][index] = x1_values[i+4]
            #choose random index to assign x2 that is not the same as x1
            index2 = np.random.randint(0, 10)
            while index2 == index:
                index2 = np.random.randint(0, 10)
            X[i][index2] = x2_values[i]
            X[i+1][index2] = x2_values[i+1]
            X[i+2][index2] = x2_values[i+2]
            X[i+3][index2] = x2_values[i+3]
            X[i+4][index2] = x2_values[i+4]
            i+=5


        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, labels,
                                                            test_size=0.2,
                                                            random_state=42)
        return X_train, X_test, y_train, y_test

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
        np.column_stack((x1_values, x2_values,x3_values)), labels, test_size=0.2,
        random_state=42)
    return X_train, X_test, y_train, y_test
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


def algo():
    # Initialize a Decision Tree classifier
    X_train, X_test, y_train, y_test = create_n_dim()
    dt_classifier = DecisionTreeClassifier()
    # Fit the classifier to the training data
    dt_classifier.fit(X_train, y_train)
    #check depth of the tree
    print(dt_classifier.get_depth())
    # Make predictions on the test data
    y_pred = dt_classifier.predict(X_test)
    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    # Get selected paths for test samples
    selected_paths = get_selected_features(dt_classifier.tree_, X_test)

    # Calculate intersection and union of selected paths
    intersection = set.intersection(*selected_paths)
    union = set.union(*selected_paths)

    print("Intersection of selected paths:", intersection)
    print("Union of selected paths:", union)




if __name__ == '__main__':
    algo()
