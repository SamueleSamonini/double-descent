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