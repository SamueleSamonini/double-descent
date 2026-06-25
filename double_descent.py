import numpy as np


def generate_data(n_train, n_test, n_features, noise_std, rng):
    """Generate a regression dataset with noisy linear labels.

    Data model:  y = X @ beta + noise
      - X has standard Gaussian entries  N(0, 1)
      - noise is Gaussian  N(0, noise_std^2)
      - 'n_features' is the ambient dimension D (maximum number of features)
      - 'beta' is the true coefficient vector, normalized to unit norm
        (so the signal variance is 1 and the signal-to-noise ratio is
         controlled only by noise_std)

    'rng' is a numpy generator (np.random.default_rng) passed from outside
    to ensure reproducibility.
    """
    # true coefficients: the same call generates both train and test
    beta = rng.standard_normal(n_features)
    beta = beta / np.linalg.norm(beta)

    # Gaussian inputs
    X_train = rng.standard_normal((n_train, n_features))
    X_test = rng.standard_normal((n_test, n_features))

    # labels = linear signal + Gaussian noise
    y_train = X_train @ beta + noise_std * rng.standard_normal(n_train)
    y_test = X_test @ beta + noise_std * rng.standard_normal(n_test)

    return X_train, y_train, X_test, y_test, beta


def fit_least_squares(X, y):
    """Least squares via the minimum-norm pseudoinverse, computed from the SVD.

    We write the Moore-Penrose pseudoinverse explicitly:
        X = U @ diag(s) @ V^T   (economy SVD)
        w = V @ diag(1/s) @ U^T @ y

    This single formula handles both regimes:
      - if d <= n it returns the ordinary least squares solution,
      - if d >  n it returns the minimum-norm solution among all the
        infinitely many vectors that interpolate the training data.

    The minimum-norm choice in the over-parameterized regime is exactly
    what produces the second descent.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # invert only the numerically non-zero singular values (avoids 1/0).
    # values that are small but non-zero are kept on purpose: near the
    # interpolation threshold they blow up the solution norm -> the peak.
    tol = np.finfo(float).eps * max(X.shape) * s.max()
    s_inv = np.where(s > tol, 1.0 / s, 0.0)

    w = Vt.T @ (s_inv * (U.T @ y))
    return w


def fit_ridge(X, y, lam):
    """Ridge regression in closed form:
        w = (X^T X + lam * I)^(-1) X^T y

    The penalty lam > 0 makes the system always invertible and the
    solution unique, even when d > n. This is our tool to attenuate
    (and eventually remove) the interpolation peak.
    """
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d)
    w = np.linalg.solve(A, X.T @ y)
    return w


def predict(X, w):
    """Linear prediction."""
    return X @ w


def mse(y_true, y_pred):
    """Mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def run_experiment(d_grid, n_train, n_test, n_features, noise_std,
                   n_trials, lam, seed):
    """Run the double descent experiment.

    For each model complexity d in 'd_grid' we:
      - fit least squares and ridge on the first d features,
      - record train MSE, test MSE and the solution norm ||w||,
    averaging everything over 'n_trials' independent datasets.

    Returns a dict of arrays (one value per d), each averaged over trials:
      'ls_train', 'ls_test', 'ls_norm', 'ridge_train', 'ridge_test', 'ridge_norm'.
    """
    n_d = len(d_grid)

    # accumulators: rows = trials, cols = complexities d
    ls_train = np.zeros((n_trials, n_d))
    ls_test = np.zeros((n_trials, n_d))
    ls_norm = np.zeros((n_trials, n_d))
    ridge_train = np.zeros((n_trials, n_d))
    ridge_test = np.zeros((n_trials, n_d))
    ridge_norm = np.zeros((n_trials, n_d))

    rng = np.random.default_rng(seed)

    for t in range(n_trials):
        # a fresh dataset for each trial (new beta, new X, new noise)
        X_train, y_train, X_test, y_test, _ = generate_data(
            n_train, n_test, n_features, noise_std, rng
        )

        for j, d in enumerate(d_grid):
            Xtr, Xte = X_train[:, :d], X_test[:, :d]

            # least squares (minimum-norm via SVD)
            w = fit_least_squares(Xtr, y_train)
            ls_train[t, j] = mse(y_train, predict(Xtr, w))
            ls_test[t, j] = mse(y_test, predict(Xte, w))
            ls_norm[t, j] = np.linalg.norm(w)

            # ridge regression
            w = fit_ridge(Xtr, y_train, lam)
            ridge_train[t, j] = mse(y_train, predict(Xtr, w))
            ridge_test[t, j] = mse(y_test, predict(Xte, w))
            ridge_norm[t, j] = np.linalg.norm(w)

    # average over trials
    return {
        "ls_train": ls_train.mean(axis=0),
        "ls_test": ls_test.mean(axis=0),
        "ls_norm": ls_norm.mean(axis=0),
        "ridge_train": ridge_train.mean(axis=0),
        "ridge_test": ridge_test.mean(axis=0),
        "ridge_norm": ridge_norm.mean(axis=0),
    }

def fit_gradient_descent(X, y, n_steps, lr=None):
    """Least squares via batch gradient descent on the squared loss.

    Started from w = 0. The key fact we want to show: in the
    over-parameterized regime (d > n), gradient descent from zero
    initialization converges to the SAME minimum-norm solution returned by
    the pseudoinverse. No explicit penalty is used, yet GD selects the
    low-norm interpolating solution: this is 'implicit regularization'.

    Loss:     L(w) = (1/n) * ||X w - y||^2
    Gradient: grad = (2/n) * X^T (X w - y)

    If 'lr' is None we set the step size from the largest eigenvalue of the
    Hessian (2/n) X^T X, which guarantees the iteration is stable.
    """
    n, d = X.shape

    if lr is None:
        # largest singular value of X -> largest eigenvalue of the Hessian
        s_max = np.linalg.norm(X, 2)
        L = (2.0 / n) * s_max ** 2
        lr = 1.0 / L

    w = np.zeros(d)
    for _ in range(n_steps):
        grad = (2.0 / n) * X.T @ (X @ w - y)
        w = w - lr * grad
    return w