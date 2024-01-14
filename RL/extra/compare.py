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


def decision_tree():
    # Splitting data into training and testing sets
    X, y, _, number_of_features = utils.load_data_labels()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = DecisionTreeClassifier(random_state=42)  # You can specify hyperparameters here
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Number of features: {number_of_features}")


def cut_decition_tree():
    X, y, _, number_of_features = utils.load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    for i in range(1, number_of_features):
        max_features_to_use = i
        clf = DecisionTreeClassifier(max_features=max_features_to_use, random_state=42)  # For classification tasks
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        print(f"Number of features: {max_features_to_use}")


def cut_decision_tree_7():
    X, y, _, number_of_features = utils.load_diabetes()
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a decision tree classifier with a maximum depth of 7 features
    tree_classifier = DecisionTreeClassifier(max_depth=7)

    # Train the decision tree classifier on the training data
    tree_classifier.fit(X_train, y_train)

    # Predict using the trained model on the test set
    y_pred = tree_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of the decision tree model: {accuracy:.4f}")
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    # selected_samples = [0, 10, 20]  # Change these indices as needed
    # for idx in selected_samples:
    #     sample = X_test[idx].reshape(1, -1)
    #     path = tree_classifier.decision_path(sample)
    #     print(f"Path of sample {idx}: {path.indices}")


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
    cut_decision_tree_7()


if __name__ == "__main__":
    os.chdir("C:\\Users\\kashann\\PycharmProjects\\RLadaptive\\RL\\")
    main()
