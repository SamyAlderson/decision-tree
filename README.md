# Decision Tree

A basic decision tree implementation in Python for data science tasks.

## What & Why

This project provides a simple decision tree implementation using Python. It's designed for data science tasks, such as classification and regression. The goal is to create a clear, easy-to-understand framework for decision tree creation and evaluation.

## Install

1. Clone the repository using `git clone https://github.com/samyalder/decision-tree.git`
2. Install the dependencies using `pip install -r requirements.txt`
3. Run the tests using `python -m unittest discover -s tests`

## Usage

Create a decision tree using the `DecisionTree` class in `src/tree.py`. Pass in your data and target variable to create a tree. You can then use the `evaluate` method to evaluate the tree's performance on a test dataset.

```python
from src.tree import DecisionTree

# Create a decision tree
tree = DecisionTree(X, y)

# Evaluate the tree's performance on a test dataset
accuracy = tree.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
```

## Build from Source

1. Clone the repository using `git clone https://github.com/samyalder/decision-tree.git`
2. Install the dependencies using `pip install -r requirements.txt`
3. Run the tests using `python -m unittest discover -s tests`

## Project Structure

```bash
.
├── README.md
├── setup.py
├── pyproject.toml
├── .gitignore
├── src
│   ├── main.py
│   ├── utils.py
│   ├── tree.py
│   └── __init__.py
├── tests
│   ├── test_tree.py
│   ├── test_utils.py
│   └── __init__.py
└── requirements.txt
```

## License

This project is licensed under the MIT License.

## Dependencies

* `numpy`
* `pandas`