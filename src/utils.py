# src/utils.py

import numpy as np
import pandas as pd

def entropy(y):
    """
    Calculate the entropy of a class distribution.

    :param y: Class labels (numpy array)
    :return: Entropy value
    """
    # Avoid division by zero
    p = np.unique(y, return_counts=True)[1] / len(y)
    return -np.sum(p * np.log2(p))

def gini_impurity(y):
    """
    Calculate the Gini impurity of a class distribution.

    :param y: Class labels (numpy array)
    :return: Gini impurity value
    """
    p = np.unique(y, return_counts=True)[1] / len(y)
    return 1 - np.sum(p ** 2)

def information_gain(X, y, feature, split):
    """
    Calculate the information gain from splitting a feature.

    :param X: Feature values (numpy array)
    :param y: Class labels (numpy array)
    :param feature: Feature index
    :param split: Split value
    :return: Information gain value
    """
    # Split the data into left and right child nodes
    left_idx = X[:, feature] <= split
    right_idx = X[:, feature] > split

    # Calculate the weighted average of the entropies
    left_entropy = entropy(y[left_idx])
    right_entropy = entropy(y[right_idx])
    return (len(left_idx) / len(y)) * left_entropy + (len(right_idx) / len(y)) * right_entropy - entropy(y)

def calculate_metrics(y_true, y_pred):
    """
    Calculate accuracy, precision, recall, and F1-score.

    :param y_true: True class labels (numpy array)
    :param y_pred: Predicted class labels (numpy array)
    :return: Accuracy, precision, recall, and F1-score values
    """
    # Calculate accuracy
    accuracy = np.sum(y_true == y_pred) / len(y_true)

    # Calculate precision and recall
    precision = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
    recall = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)

    # Calculate F1-score
    f1 = 2 * precision * recall / (precision + recall)

    return accuracy, precision, recall, f1