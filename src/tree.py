# Decision tree implementation

class Node:
    """Decision tree node"""
    def __init__(self, feature=None, value=None, left=None, right=None, *, label=None):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.label = label

class DecisionTree:
    """Decision tree implementation"""
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def _split(self, data, feature, value):
        """Split data based on feature and value"""
        mask = data[:, feature] <= value
        left = data[mask]
        right = data[~mask]
        return left, right

    def _gain(self, data, feature):
        """Calculate information gain for feature"""
        total = len(data)
        values, counts = np.unique(data[:, feature], return_counts=True)
        entropy = -np.sum((counts / total) * np.log2(counts / total))
        return entropy

    def _best_split(self, data):
        """Find best split for data"""
        n_features = data.shape[1]
        best_feature = None
        best_value = None
        best_gain = -1
        for feature in range(n_features):
            values = np.unique(data[:, feature])
            for value in values:
                left, right = self._split(data, feature, value)
                gain = self._gain(left, feature) + self._gain(right, feature)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_value = value
        return best_feature, best_value

    def _build_tree(self, data, depth=0):
        """Build decision tree"""
        if depth >= self.max_depth:
            return Node(label=np.argmax(np.mean(data, axis=0), axis=1))
        feature, value = self._best_split(data)
        if feature is None:
            return Node(label=np.argmax(np.mean(data, axis=0), axis=1))
        left, right = self._split(data, feature, value)
        left_node = self._build_tree(left, depth + 1)
        right_node = self._build_tree(right, depth + 1)
        return Node(feature, value, left_node, right_node)

    def fit(self, data):
        """Fit decision tree to data"""
        self.root = self._build_tree(data)

    def predict(self, data):
        """Predict labels for data"""
        stack = [(self.root, data)]
        labels = []
        while stack:
            node, data = stack.pop()
            if node.label is not None:
                labels.append(node.label)
            else:
                left, right = self._split(data, node.feature, node.value)
                stack.append((node.left, left))
                stack.append((node.right, right))
        return np.array(labels)

import numpy as np