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


def algo():
    # Initialize a Decision Tree classifier
    X_train, X_test, y_train, y_test = crete_data()
    dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=3)
    # Fit the classifier to the training data
    dt_classifier.fit(X_train, y_train)
    # Make predictions on the test data
    y_pred = dt_classifier.predict(X_test)
    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    # Plot the decision boundary and the tree
    # plt.figure(figsize=(10, 6))
    # plot_tree(dt_classifier, filled=True, feature_names=['X', 'Y'], class_names=['0', '1'])
    plt.show()


if __name__ == '__main__':
    algo()
