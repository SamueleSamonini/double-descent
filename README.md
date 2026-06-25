# Double Descent in Linear Models

Experimental project for the *Statistical Methods for Machine Learning* course
(A.Y. 2025/26).

This project empirically reproduces the **double descent** risk curve in the
setting of linear regression, following Belkin et al. (2019), *Reconciling
modern machine learning practice and the classical bias–variance trade-off*
(PNAS).

## Background

The classical bias–variance trade-off predicts a U-shaped test-error curve:
increasing model complexity first improves generalization, then hurts it once
the model starts to overfit. Modern over-parameterized models contradict this
picture, achieving low test error even when they interpolate the training data.
Belkin et al. reconcile the two views with a single **double descent** curve:
beyond the *interpolation threshold* (the point where the model first fits the
training data exactly), the test error descends a second time, often below the
classical sweet spot.

## What this project does

We keep the number of training points `n` fixed and vary the number of features
`d`, so that the interpolation threshold is crossed exactly at `d = n`. On a
synthetic regression dataset with noisy linear labels, we:

- implement **ordinary least squares** (via the minimum-norm pseudoinverse) and
  **ridge regression**, and measure training error, test error, and the norm of
  the learned solution as a function of `d`;
- reproduce the double descent curve and identify the interpolation threshold;
- show that ridge regularization attenuates and eventually removes the peak.

Two extensions are included:

- **Gradient descent vs closed form** — gradient descent from zero
  initialization recovers the minimum-norm solution (implicit regularization);
- **Effect of noise** — the height of the interpolation peak grows with the
  label noise.

All algorithms are implemented **from scratch** with `numpy`; no machine-learning
library is used to fit the models. Results are fully reproducible from a fixed
random seed.

## Repository structure

- `src/double_descent.py` — core functions: data generation and estimators
- `notebooks/01_main_experiment.ipynb` — main experiment and report figures
- `notebooks/02_extensions.ipynb` — the two extensions
- `report/report.pdf` — final report
- `report/figures/` — figures used in the report
- `requirements.txt` — Python dependencies

## How to run

1. (Optional) Create and activate a virtual environment.
2. Install the dependencies: `pip install -r requirements.txt`
3. Launch Jupyter (`jupyter notebook`) and open the notebooks in the `notebooks/`
   folder. Run all cells from top to bottom, starting with
   `01_main_experiment.ipynb`, then `02_extensions.ipynb`. The notebooks import
   the core module from `../src`, so they should be run from their location
   inside `notebooks/`.

## Report

The full write-up, including methodology, results, and a discussion of the
positive and negative aspects of the work, is available in `report/report.pdf`.

## Reference

M. Belkin, D. Hsu, S. Ma, and S. Mandal. *Reconciling modern machine learning
practice and the classical bias–variance trade-off*. Proceedings of the National
Academy of Sciences (PNAS), 116(32):15849–15854, 2019.

## Author

Samuele Samonini — Student ID: 73427A