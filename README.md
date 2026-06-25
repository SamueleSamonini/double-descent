# Double Descent in Linear Models

Experimental project for the *Statistical Methods for Machine Learning* course
(A.Y. 2025/26). The project empirically reproduces the **double descent** risk
curve in the setting of linear regression, following Belkin et al. (2019),
*Reconciling modern machine learning practice and the classical bias–variance
trade-off* (PNAS).

## Overview

We keep the number of training points `n` fixed and vary the number of features
`d`, so that the interpolation threshold is crossed exactly at `d = n`. We study
ordinary least squares and ridge regression, and we analyze training error, test
error and the norm of the learned solution. Two extensions are included:
gradient descent versus the closed-form solution (implicit regularization), and
the effect of label noise on the interpolation peak.

All algorithms are implemented **from scratch** with `numpy`; no machine-learning
library is used to fit the models.

## Repository structure

- `double_descent.py` — core functions (data generation and estimators).
- `01_main_experiment.ipynb` — main experiment and report figures.
- `02_extensions.ipynb` — the two extensions.
- `figures/` — figures used in the report.
- `report.pdf` — final report (PDF).
- `requirements.txt` — Python dependencies.

## How to run

1. (Optional) Create and activate a virtual environment.
2. Install the dependencies: