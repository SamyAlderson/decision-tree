# tests/test_utils.py

import unittest
import numpy as np
import pandas as pd
from src.utils import split_data, evaluate_accuracy

class TestUtils(unittest.TestCase):
    def test_split_data(self):
        # Test splitting data with a single feature
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        feature_index = 0
        expected_X_train = np.array([[1], [3]])
        expected_X_test = np.array([[5]])
        expected_y_train = np.array([0, 1])
        expected_y_test = np.array([0])
        X_train, X_test, y_train, y_test = split_data(X, y, feature_index)
        self.assertTrue(np.array_equal(X_train, expected_X_train))
        self.assertTrue(np.array_equal(X_test, expected_X_test))
        self.assertTrue(np.array_equal(y_train, expected_y_train))
        self.assertTrue(np.array_equal(y_test, expected_y_test))

        # Test splitting data with multiple features
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = np.array([0, 1, 0])
        feature_index = 1
        expected_X_train = np.array([[1, 2], [4, 5]])
        expected_X_test = np.array([[7, 8]])
        expected_y_train = np.array([0, 1])
        expected_y_test = np.array([0])
        X_train, X_test, y_train, y_test = split_data(X, y, feature_index)
        self.assertTrue(np.array_equal(X_train, expected_X_train))
        self.assertTrue(np.array_equal(X_test, expected_X_test))
        self.assertTrue(np.array_equal(y_train, expected_y_train))
        self.assertTrue(np.array_equal(y_test, expected_y_test))

    def test_evaluate_accuracy(self):
        # Test evaluating accuracy with correct predictions
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1, 0])
        accuracy = evaluate_accuracy(y_true, y_pred)
        self.assertEqual(accuracy, 1.0)

        # Test evaluating accuracy with incorrect predictions
        y_true = np.array([0, 1, 0])
        y_pred = np.array([1, 0, 1])
        accuracy = evaluate_accuracy(y_true, y_pred)
        self.assertEqual(accuracy, 0.0)

        # Test evaluating accuracy with different number of classes
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        accuracy = evaluate_accuracy(y_true, y_pred)
        self.assertEqual(accuracy, 1.0)

if __name__ == '__main__':
    unittest.main()