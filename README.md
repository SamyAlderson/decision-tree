# Decision Tree
A simple Python implementation of a decision tree for data science tasks

## Installation

To get started, you'll need Python 3.8+ and the `numpy` library. You can install the latter using pip:
```bash
pip install numpy
```
Now you can install the `decision-tree` package from source using pip:
```bash
pip install .
```
## Usage

The `decision_tree` module exports a `DecisionTree` class that you can use to train and evaluate your decision trees. Here's an example:
```python
from decision_tree import DecisionTree

# Create a decision tree
tree = DecisionTree()

# Train the tree on some data
tree.fit(X, y)

# Evaluate the tree on some test data
accuracy = tree.score(X_test, y_test)
print(accuracy)
```
## Build from Source

If you want to build the project from source, simply run the following commands:
```bash
git clone https://github.com/SamyAlderson/decision-tree.git
cd decision-tree
pip install .
```
## Project Structure

* `decision_tree.py`: The main module that exports the `DecisionTree` class
* `tests.py`: A test suite that covers the `DecisionTree` class
* `utils.py`: A utility module that contains some helper functions
* `data`: A directory that contains some example data
* `requirements.txt`: A file that lists the project's dependencies

## License

Copyright (c) 2026 SamyAlderson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.