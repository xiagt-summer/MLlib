import numpy as np

def train(
    X,
    y,
    penalty=None,          # None | 'l2' | 'l1'
    lam: float = 1e-3,
    sample_weight=None,    # optional (N,)
    max_iter: int = 1000,  # used for L1 (coordinate descent)
    tol: float = 1e-8,     # used for L1: parameter convergence (L1-norm)
    verbose: bool = False,
):
    """
    General linear regression with optional L2 (Ridge) or L1 (Lasso) regularization.
    Intercept (first column of X == 1.0) is automatically detected and NOT regularized.

    Objective (weighted):
      For None:  minimize  (1/(2*sum_w)) * sum_i w_i * (y_i - x_i^T b)^2
      For L2:    same + (lam/2) * ||b_no_intercept||_2^2
      For L1:    same + lam * ||b_no_intercept||_1

    Parameters
    ----------
    X : (N, D) array-like
        Feature matrix. If you want an intercept, include a column of ones as X[:,0].
    y : (N,) array-like
        Target vector.
    penalty : {None, 'l2', 'l1'}, default=None
        Regularization type.
    lam : float, default=1e-3
        Regularization strength. Intercept is never penalized.
    sample_weight : (N,) array-like or None
        Optional non-negative sample weights.
    max_iter : int, default=1000
        Max iterations for L1 coordinate descent (unused for None/L2).
    tol : float, default=1e-8
        Convergence tolerance on parameter change (L1 uses L1-norm).
    verbose : bool, default=False
        If True, print brief progress for L1 optimization.

    Returns
    -------
    beta : (D,) ndarray
        Estimated coefficients.
    y_hat : (N,) ndarray
        In-sample predictions X @ beta.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    N, D = X.shape

    # sample weights
    if sample_weight is None:
        w = np.ones(N, dtype=float)
    else:
        w = np.asarray(sample_weight, dtype=float).clip(min=0.0)
    sw_sum = w.sum() + 1e-12  # avoid div by zero

    # detect intercept: first column equals 1.0 (within tolerance)
    has_intercept = np.allclose(X[:, 0], 1.0)
    penalize_mask = np.ones(D, dtype=bool)
    if has_intercept:
        penalize_mask[0] = False  # do not regularize intercept

    pen = (penalty.lower() if isinstance(penalty, str) else None)

    # ---------- None: ordinary (weighted) least squares via lstsq ----------
    if pen is None:
        # Weighted least squares can be solved by sqrt-weighted augmentation
        sqrtw = np.sqrt(w)
        Xw = X * sqrtw[:, None]
        yw = y * sqrtw
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        y_hat = X @ beta
        return beta, y_hat

    # ---------- L2: Ridge via Tikhonov augmentation + lstsq ----------
    if pen == 'l2':
        # Build penalization matrix E of shape (D, D) with 1 on penalized diagonals, 0 otherwise
        E = np.diag(penalize_mask.astype(float))
        # sqrt(λ) * E for Tikhonov
        # Weighted least squares with augmentation:
        #   minimize || sqrt(W)*(y - Xb) ||^2 + || sqrt(λ) E b ||^2
        sqrtw = np.sqrt(w)
        Xw = X * sqrtw[:, None]
        yw = y * sqrtw
        X_aug = np.vstack([Xw, np.sqrt(lam) * E])
        y_aug = np.concatenate([yw, np.zeros(D, dtype=float)])
        beta, *_ = np.linalg.lstsq(X_aug, y_aug, rcond=None)
        y_hat = X @ beta
        return beta, y_hat

    # ---------- L1: Lasso via Coordinate Descent (weighted) ----------
    if pen == 'l1':
        # Initialize
        beta = np.zeros(D, dtype=float)
        # Precompute weighted column norms
        # c_j = sum_i w_i * x_ij^2
        c = (w[:, None] * (X ** 2)).sum(axis=0) + 1e-18  # avoid /0
        # Residual r = y - Xb
        r = y - X @ beta

        lam_eff = lam * sw_sum  # threshold aligns with objective scaling

        def soft_threshold(a, t):
            # sign(a) * max(|a| - t, 0)
            return np.sign(a) * np.maximum(np.abs(a) - t, 0.0)

        for it in range(max_iter):
            beta_old = beta.copy()

            # --- update intercept (no penalty) ---
            if has_intercept:
                j = 0
                # a0 = sum_i w_i * x_i0 * (r_i + x_i0 * beta0). With x_i0 == 1
                a0 = (w * (r + X[:, j] * beta[j])).sum()
                beta[j] = a0 / c[j]
                # update residual
                r = y - X @ beta

            # --- update other coefficients with L1 penalty ---
            for j in range(1 if has_intercept else 0, D):
                # a_j = sum_i w_i * x_ij * (r_i + x_ij * beta_j)
                a_j = (w * X[:, j] * (r + X[:, j] * beta[j])).sum()
                # closed-form CD update with soft-thresholding
                b_new = soft_threshold(a_j, lam_eff) / c[j]
                if b_new != beta[j]:
                    beta[j] = b_new
                    # incremental residual update:
                    # r_new = r_old - x_j * (b_new - b_old)
                    r -= X[:, j] * (b_new - (beta[j] if False else 0))  # but we already set beta[j]=b_new
                    # 简化：直接重算更安全（对小 D 开销可忽略）
                    r = y - X @ beta

            # convergence check
            if np.linalg.norm(beta - beta_old, ord=1) <= tol * (1.0 + np.linalg.norm(beta_old, ord=1)):
                if verbose:
                    print(f"[L1-CD] converged at iter {it}")
                break

        y_hat = X @ beta
        return beta, y_hat

    raise ValueError("penalty must be None, 'l2', or 'l1'")

def predict(X, beta):
    """
    Predict target values using a fitted linear_regression model.

    Parameters
    ----------
    X : (N, D) array-like
        Feature matrix. Should match the feature layout used in training
        (including intercept column if it was used).
    beta : (D,) array-like
        Coefficient vector returned by linear_regression().

    Returns
    -------
    y_hat : (N,) ndarray
        Predicted target values.
    """
    X = np.asarray(X, dtype=float)
    beta = np.asarray(beta, dtype=float)
    return X @ beta