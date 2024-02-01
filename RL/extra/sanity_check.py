import os

import numpy as np
import matplotlib.pyplot as plt
from nltk import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_classification



def algo():
    # Initialize a Decision Tree classifier
    X, y = create_sanity_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    dt_classifier = DecisionTreeClassifier(max_depth=14)

    # Fit the classifier to the training data
    dt_classifier.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = dt_classifier.predict(X_test)

    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')




def create_sanity_data():
    # Set the number of data points
    n_samples = 100
    num_total_features = 30

    # Generate synthetic data with two features
    np.random.seed(42)
    total_points = np.empty((0, num_total_features))
    labels = np.empty((0, 1))
    for i in range(num_total_features - 1):
        for j in range(i + 1, num_total_features):
            X = np.random.uniform(0, 100, size=(n_samples, num_total_features))
            # Create a diagonal decision boundary
            y = (X[:, i] > X[:, j]).astype(int).reshape(-1, 1)
            total_points = np.vstack((total_points, X))
            labels = np.vstack((labels, y))

    return total_points, labels


if __name__ == '__main__':
    algo()
    dir= "/RL"
    os.chdir(dir)
    X,y = create_sanity_data()
    #concatenate the labels and the data
    data = np.concatenate((X,y),axis=1)
    #save x to csv file
    np.savetxt("data.csv", data, delimiter=",")

