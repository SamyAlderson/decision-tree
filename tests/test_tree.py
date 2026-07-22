# Unit tests for decision tree functionality
import unittest
from src.tree import DecisionTree
import numpy as np
import pandas as pd

class TestDecisionTree(unittest.TestCase):

    def test_create_tree(self):
        # Create a sample dataset
        data = {
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [6, 7, 8, 9, 10],
            'target': [1, 0, 1, 0, 1]
        }
        df = pd.DataFrame(data)

        # Create a decision tree
        tree = DecisionTree(df, 'target', ['feature1', 'feature2'])

        # Check that the tree is created
        self.assertIsNotNone(tree.root)

    def test_train_tree(self):
        # Create a sample dataset
        data = {
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [6, 7, 8, 9, 10],
            'target': [1, 0, 1, 0, 1]
        }
        df = pd.DataFrame(data)

        # Create a decision tree
        tree = DecisionTree(df, 'target', ['feature1', 'feature2'])

        # Train the tree
        tree.train_tree()

        # Check that the tree is trained
        self.assertIsNotNone(tree.root)

    def test_predict_tree(self):
        # Create a sample dataset
        data = {
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [6, 7, 8, 9, 10],
            'target': [1, 0, 1, 0, 1]
        }
        df = pd.DataFrame(data)

        # Create a decision tree
        tree = DecisionTree(df, 'target', ['feature1', 'feature2'])

        # Train the tree
        tree.train_tree()

        # Create a test sample
        test_sample = {'feature1': 3, 'feature2': 8}

        # Make a prediction
        prediction = tree.predict(test_sample)

        # Check that the prediction is correct
        self.assertEqual(prediction, 1)

    def test_evaluate_tree(self):
        # Create a sample dataset
        data = {
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [6, 7, 8, 9, 10],
            'target': [1, 0, 1, 0, 1]
        }
        df = pd.DataFrame(data)

        # Create a decision tree
        tree = DecisionTree(df, 'target', ['feature1', 'feature2'])

        # Train the tree
        tree.train_tree()

        # Evaluate the tree
        evaluation = tree.evaluate_tree()

        # Check that the evaluation is correct
        self.assertEqual(evaluation, {'accuracy': 0.8, 'precision': 0.75, 'recall': 0.8, 'f1_score': 0.8})

if __name__ == '__main__':
    unittest.main()