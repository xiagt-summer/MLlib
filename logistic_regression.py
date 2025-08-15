import numpy as np

# ============================ utilities ============================

def _sigmoid_stable(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid; split by sign to avoid overflow."""
    p = np.empty_like(z, dtype=float)
    pos = z >= 0
    p[pos]  = 1.0 / (1.0 + np.exp(-z[pos]))
    ez      = np.exp(z[~pos])
    p[~pos] = ez / (1.0 + ez)
    return p

# ============================ core: train ============================

def train(
    X,
    y,
    classes=None,           # order of labels; if None -> np.unique(y)
    penalty=None,           # None | 'l2' | 'l1'
    lam: float = 1e-3,      # regularization strength
    max_iter: int = 100,    # max outer iterations
    tol: float = 1e-6,      # convergence tolerance on params
    sample_weight=None,     # per-sample nonnegative weights
):
    """
    Binary logistic regression (sigmoid), API-aligned with softmax_regression.

    Parameters
    ----------
    X : (N, D) array-like
        Feature matrix. If the first column is all 1.0, it is treated as an intercept
        and is not regularized.
    y : (N,) array-like
        Labels; mapped to {0,1} by `classes`.
    classes : (2,) array-like or None, optional
        Label order. If None, uses sorted unique labels from `y`.
        Returned probabilities follow this order.
    penalty : {None, 'l2', 'l1'}, optional
        Regularization type: no penalty, ridge (IRLS), or lasso (coordinate descent).
    lam : float, optional
        Regularization strength for L2/L1. Ignored when `penalty is None`.
    max_iter : int, optional
        Maximum number of IRLS/coordinate-descent sweeps.
    tol : float, optional
        Parameter-change tolerance (L2 for IRLS, L1 for lasso).
    sample_weight : (N,) array-like or None, optional
        Nonnegative sample weights. Defaults to uniform weights.

    Returns
    -------
    W : (D, 2) ndarray
        Two-column weights compatible with softmax-style helpers.
        Column 0 (for `classes[0]`) is all zeros (reference); column 1 is the learned vector.
    P : (N, 2) ndarray
        Probabilities per class in `classes` order: [P(class0), P(class1)].
    classes : (2,) ndarray
        The label order used.
    """
    # inputs / mapping
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    N, D = X.shape

    if classes is None:
        classes = np.unique(y)
    classes = np.asarray(classes)
    if len(classes) != 2:
        raise ValueError("This binary logistic solver requires exactly 2 classes.")
    y_map = {classes[0]: 0, classes[1]: 1}
    try:
        yb = np.vectorize(y_map.__getitem__)(y).astype(float)
    except KeyError as e:
        raise ValueError(f"Label {e.args[0]} not in `classes`: {classes!r}")

    # sample weights
    sw = np.ones(N, dtype=float) if sample_weight is None else np.asarray(sample_weight, float).clip(min=0.0)

    # intercept detection (do not regularize if X[:,0] is all 1.0)
    no_pen_intercept = np.allclose(X[:, 0], 1.0)

    # train one weight vector w for classes[1] vs classes[0]
    w = np.zeros(D, dtype=float)
    pen = (penalty.lower() if isinstance(penalty, str) else None)

    if pen in (None, 'l2'):
        for _ in range(max_iter):
            z  = X @ w
            p  = _sigmoid_stable(z)
            pj = p * (1.0 - p)
            Wd = sw * pj  # IRLS diagonal

            if np.all(Wd < 1e-12):
                break

            z_tilde = z + (yb - p) / np.clip(pj, 1e-12, None)

            # ridge matrix
            if pen is None or lam <= 0.0:
                R = 0.0
            else:
                R = np.diag([0.0] + [lam] * (D - 1)) if no_pen_intercept else lam * np.eye(D)

            Xw = X * Wd[:, None]
            H  = X.T @ Xw + (0.0 if isinstance(R, float) else R)
            g  = X.T @ (Wd * z_tilde)

            w_new = np.linalg.solve(H, g)
            if np.linalg.norm(w_new - w) <= tol * (1.0 + np.linalg.norm(w)):
                w = w_new
                break
            w = w_new

    elif pen == 'l1':
        for _ in range(max_iter):
            w_old = w.copy()
            z  = X @ w
            p  = _sigmoid_stable(z)
            r  = yb - p
            pj = p * (1 - p)

            # cyclic coordinate descent with soft-thresholding
            for j in range(D):
                if j == 0 and no_pen_intercept:
                    grad = np.sum(sw * r)
                    hess = np.sum(sw * pj)
                    if hess > 1e-12:
                        w[j] += grad / hess
                else:
                    xj   = X[:, j]
                    grad = np.sum(sw * xj * r)
                    hess = np.sum(sw * pj * (xj ** 2))
                    if hess > 1e-12:
                        z_j  = w[j] + grad / hess
                        w[j] = np.sign(z_j) * max(abs(z_j) - lam / hess, 0.0)

                # refresh residuals
                z  = X @ w
                p  = _sigmoid_stable(z)
                r  = yb - p
                pj = p * (1 - p)

            if np.linalg.norm(w - w_old, ord=1) <= tol:
                break
    else:
        raise ValueError("penalty must be None, 'l2', or 'l1'")

    # package like softmax_regression
    W = np.zeros((D, 2), dtype=float)
    W[:, 1] = w
    P = predict_proba(X, W)
    return W, P, classes

# ====================== prediction helpers ======================

def predict_proba_sigmoid(X, W):
    """
    Parameters
    ----------
    X : (N, D) array-like
        Feature matrix (same columns as during training).
    W : (D, 2) ndarray
        Weight matrix from `train`.

    Returns
    -------
    p1 : (N,) ndarray
        P(y == classes[1]) via sigmoid(X @ W[:, 1]).
    """
    X = np.asarray(X, dtype=float)
    z = X @ W[:, 1]
    return _sigmoid_stable(z)

def predict_proba(X, W):
    """
    Parameters
    ----------
    X : (N, D) array-like
        Feature matrix.
    W : (D, 2) ndarray
        Weight matrix.

    Returns
    -------
    P : (N, 2) ndarray
        [P(class0), P(class1)] in the same `classes` order used at training.
    """
    p1 = predict_proba_sigmoid(X, W)
    return np.column_stack([1.0 - p1, p1])

def predict_class(X, W, classes):
    """
    Parameters
    ----------
    X : (N, D) array-like
        Feature matrix.
    W : (D, 2) ndarray
        Weight matrix.
    classes : (2,) array-like
        Label order from training.

    Returns
    -------
    y_pred : (N,) ndarray
        Predicted labels from `classes`.
    """
    P = predict_proba(X, W)
    idx = np.argmax(P, axis=1)
    return np.asarray(classes)[idx]
