# MLlib

A simple machine learning library implementation in NumPy.

## Algorithms

### Linear Regression
- Ordinary least squares
- Ridge regression (L2 regularization)
- Lasso regression (L1 regularization)
- Support for sample weights
- Automatic intercept detection

### Logistic Regression
- Binary classification
- L2 regularization (IRLS solver)
- L1 regularization (coordinate descent)
- Custom class labels
- Sample weighting

### Softmax Regression
- Multiclass classification
- Newton-CG solver for L2/no penalty
- Proximal gradient for L1
- Hessian-free optimization
- Weighted samples

## Usage

### Linear Regression
```python
import linear_regression

# Basic usage
beta, y_hat = linear_regression.train(X, y)

# With L2 regularization
beta, y_hat = linear_regression.train(X, y, penalty='l2', lam=0.1)

# Prediction
y_pred = linear_regression.predict(X_test, beta)
```

### Logistic Regression
```python
import logistic_regression

# Binary classification
W, P, classes = logistic_regression.train(X, y)

# Prediction
probs = logistic_regression.predict_proba(X_test, W)
labels = logistic_regression.predict_class(X_test, W, classes)
```

### Softmax Regression
```python
import softmax_regression

# Multiclass classification
W, P, classes = softmax_regression.train(X, y)

# Prediction
probs = softmax_regression.predict_proba(X_test, W)
labels = softmax_regression.predict_class(X_test, W, classes)
```

## Requirements

- NumPy

