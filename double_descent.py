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