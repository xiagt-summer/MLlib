import numpy as np

# ============================ utilities ============================

def _softmax_stable(Z: np.ndarray) -> np.ndarray:
    """Row-wise numerically stable softmax."""
    Zs = Z - Z.max(axis=1, keepdims=True)
    EZ = np.exp(Zs)
    return EZ / (EZ.sum(axis=1, keepdims=True) + 1e-12)

def _one_hot(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """One-hot encode labels y according to given class order."""
    idx = {c: i for i, c in enumerate(classes)}
    Yh = np.zeros((y.shape[0], len(classes)), dtype=float)
    for r, v in enumerate(y):
        Yh[r, idx[v]] = 1.0
    return Yh

# ====================== core: softmax regression ======================

def train(
    X,
    y,
    classes=None,
    penalty=None,            # None | 'l2' | 'l1'
    lam: float = 1e-3,
    max_iter: int = 50,
    tol: float = 1e-6,
    # Newton-CG (for None/L2)
    cg_max_iter: int = 100,
    cg_tol: float = 1e-6,
    line_search: bool = True,
    ls_beta: float = 0.5,
    ls_c1: float = 1e-4,
    # Prox (for L1)
    lr: float = 1.0,
    backtracking: bool = True,
    # misc
    W_init=None,
    sample_weight=None,
):
    """
    Multinomial logistic regression (softmax + cross-entropy).

    Optimizer:
      - penalty in {None, 'l2'} : Newton-CG (Hessian-free) + Armijo line search
      - penalty == 'l1'         : Proximal gradient (soft-threshold) + backtracking

    Intercept handling:
      - If X[:,0] is a constant 1.0 column, the corresponding weight row W[0,:]
        is treated as an intercept and is NOT regularized.

    Parameters
    ----------
    X : (N, D) array-like
        Feature matrix (include a 1.0 column if you want an intercept).
    y : (N,) array-like
        Multiclass labels. Can be any dtype; will be mapped to 'classes'.
    classes : array-like or None
        Class order. If None, uses np.unique(y) (sorted).
    penalty : {None, 'l2', 'l1'}
        Regularization type.
    lam : float
        Regularization strength.
    max_iter : int
        Max outer iterations.
    tol : float
        Convergence tolerance on parameter change (Frobenius norm).
    cg_max_iter : int
        Max iterations for inner CG (Newton-CG).
    cg_tol : float
        Tolerance for inner CG residual.
    line_search : bool
        Use Armijo backtracking in Newton-CG.
    ls_beta : float
        Step shrink factor in line search (0 < beta < 1).
    ls_c1 : float
        Armijo sufficient decrease coefficient.
    lr : float
        Initial step size for proximal gradient (L1).
    backtracking : bool
        Backtracking for proximal gradient (L1).
    W_init : (D, K) array-like or None
        Warm start for weights.
    sample_weight : (N,) array-like or None
        Optional sample weights (>=0). If provided, loss is weighted average.

    Returns
    -------
    W : (D, K) float ndarray
        Weight matrix; column k corresponds to classes[k].
    P : (N, K) float ndarray
        Predicted probabilities via softmax.
    classes : (K,) ndarray
        The class labels in the same order as columns of W.
    """
    # ---- inputs ----
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    N, D = X.shape

    if classes is None:
        classes = np.unique(y)
    classes = np.asarray(classes)
    K = len(classes)
    Yh = _one_hot(y, classes)  # (N, K)

    # sample weights
    if sample_weight is None:
        sw = np.ones(N, dtype=float)
    else:
        sw = np.asarray(sample_weight, dtype=float).clip(min=0.0)
    sw_sum = sw.sum() + 1e-12  # avoid /0

    # detect intercept row (do not regularize it)
    no_pen_intercept = np.allclose(X[:, 0], 1.0)

    # init weights
    W = np.zeros((D, K), dtype=float) if W_init is None else np.asarray(W_init, dtype=float).copy()

    # ---------- common closures ----------
    def loss_grad(Wm):
        """Weighted cross-entropy + L2 gradient (if any)."""
        Z = X @ Wm
        Pm = _softmax_stable(Z)                # (N, K)
        ce = -np.sum(sw[:, None] * Yh * np.log(Pm + 1e-12)) / sw_sum

        G = (X.T @ (sw[:, None] * (Pm - Yh))) / sw_sum  # (D, K)

        if penalty and penalty.lower() == 'l2' and lam > 0.0:
            Wpen = Wm.copy()
            if no_pen_intercept:
                Wpen[0, :] = 0.0
            ce += 0.5 * lam * np.sum(Wpen * Wpen)
            G  += lam * Wpen
        return ce, Pm, G

    def Hv(Wm, V):
        """Hessian-vector product for multinomial loss with L2 (weighted)."""
        Z = X @ Wm
        Pm = _softmax_stable(Z)                # (N, K)
        U  = X @ V                             # (N, K)
        # Local Fisher product per sample: (diag(P)-P P^T) U
        s = (U * Pm).sum(axis=1, keepdims=True)  # (N,1)
        FU = (U - s) * Pm                        # (N,K)
        Hv_mat = (X.T @ (sw[:, None] * FU)) / sw_sum  # (D,K)
        if penalty and penalty.lower() == 'l2' and lam > 0.0:
            Vpen = V.copy()
            if no_pen_intercept:
                Vpen[0, :] = 0.0
            Hv_mat += lam * Vpen
        return Hv_mat

    def flatten(M): return M.ravel()
    def unflat(v):  return v.reshape(D, K)

    # ---------- solver selection ----------
    pen = (penalty.lower() if isinstance(penalty, str) else None)

    if pen in (None, 'l2'):
        # ===== Newton-CG (Hessian-free) =====
        for _ in range(max_iter):
            L_prev, P, G = loss_grad(W)
            gnorm = np.linalg.norm(G)
            if gnorm <= tol * (1.0 + np.linalg.norm(W)):
                break

            # CG to approx solve H d = -G
            b = -flatten(G)
            x = np.zeros_like(b)
            r = b.copy()
            p = r.copy()
            rr_old = r @ r
            if rr_old >= cg_tol**2:
                for _ in range(cg_max_iter):
                    Hp = flatten(Hv(W, unflat(p)))
                    denom = p @ Hp + 1e-18
                    alpha = rr_old / denom
                    x += alpha * p
                    r -= alpha * Hp
                    rr_new = r @ r
                    if rr_new < cg_tol**2:
                        break
                    beta = rr_new / (rr_old + 1e-18)
                    p = r + beta * p
                    rr_old = rr_new
            d = unflat(x)  # Newton direction

            # Line search (Armijo)
            step = 1.0
            if line_search:
                gd = (G * d).sum()  # should be negative typically
                while True:
                    W_try = W + step * d
                    L_try, _, _ = loss_grad(W_try)
                    if L_try <= L_prev + ls_c1 * step * gd:
                        W = W_try
                        break
                    step *= ls_beta
                    if step < 1e-12:
                        W = W_try
                        break
            else:
                W = W + step * d

            # param convergence
            if np.linalg.norm(d) <= tol * (1.0 + np.linalg.norm(W)):
                break

        P = _softmax_stable(X @ W)
        return W, P, classes

    elif pen == 'l1':
        # ===== Proximal gradient (L1) =====
        # L(W) = CE(W) + lam*||W_no_intercept||_1
        def loss_only(Wm):
            Z = X @ Wm
            Pm = _softmax_stable(Z)
            ce = -np.sum(sw[:, None] * Yh * np.log(Pm + 1e-12)) / sw_sum
            if lam > 0.0:
                Wpen = Wm.copy()
                if no_pen_intercept:
                    Wpen[0, :] = 0.0
                ce += lam * np.sum(np.abs(Wpen))
            return ce

        for _ in range(max_iter):
            L_prev, P, G = loss_grad(W)  # gradient of CE only (no L1 in G)
            W_old = W.copy()

            if backtracking:
                step = lr
                # Armijo using ||G||^2 proxy
                Gsq = np.sum(G * G)
                while True:
                    W_tmp = W - step * G
                    # soft-threshold (skip intercept row)
                    if no_pen_intercept:
                        W_new = W_tmp.copy()
                        W_new[1:, :] = np.sign(W_tmp[1:, :]) * np.maximum(
                            np.abs(W_tmp[1:, :]) - step * lam, 0.0
                        )
                    else:
                        W_new = np.sign(W_tmp) * np.maximum(np.abs(W_tmp) - step * lam, 0.0)

                    L_try = loss_only(W_new)
                    if L_try <= L_prev - ls_c1 * step * Gsq:
                        W = W_new
                        break
                    step *= ls_beta
                    if step < 1e-16:
                        W = W_new
                        break
            else:
                # fixed-step prox
                W_tmp = W - lr * G
                if no_pen_intercept:
                    W = W_tmp.copy()
                    W[1:, :] = np.sign(W_tmp[1:, :]) * np.maximum(np.abs(W_tmp[1:, :]) - lr * lam, 0.0)
                else:
                    W = np.sign(W_tmp) * np.maximum(np.abs(W_tmp) - lr * lam, 0.0)

            # param convergence
            if np.linalg.norm(W - W_old) <= tol * (1.0 + np.linalg.norm(W_old)):
                break

        P = _softmax_stable(X @ W)
        return W, P, classes

    else:
        raise ValueError("penalty must be None, 'l2', or 'l1'")

# ====================== prediction helpers ======================

def predict_proba(X, W):
    """Return softmax probabilities for given data and weight matrix."""
    X = np.asarray(X, dtype=float)
    return _softmax_stable(X @ W)

def predict_class(X, W, classes):
    """Return class argmax predictions."""
    P = predict_proba(X, W)
    idx = np.argmax(P, axis=1)
    return np.asarray(classes)[idx]
