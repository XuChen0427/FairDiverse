"""
Implementation of the ICDE 2019 paper
iFair_module: Learning Individually Fair Data Representations for Algorithmic Decision Making
__url__: https://ieeexplore.ieee.org/abstract/document/8731591
__author__: Preethi Lahoti
__email__: plahoti@mpi-inf.mpg.de
"""

import numpy as np
import os
from scipy import linalg
from sklearn.metrics import pairwise
from ..probabilistic_mapping_helpers import compute_X_hat
def compute_reconstruction_loss(x_orig, x_hat):
    """Computes the reconstruction loss (L_x)."""
    return np.mean((x_hat - x_orig) ** 2)

def compute_individual_fairness_loss(D_X_f, X_hat):
    """Computes the individual fairness loss (L_z)."""
    D_X_f_hat = pairwise.euclidean_distances(X_hat, X_hat)
    L_z = linalg.norm(D_X_f - D_X_f_hat)
    return L_z


def iFair_optimisation(params, X,D_X_f=None, k=10, A_x=1e-4, A_z=1e-4, logs_path=""):
    """iFair optimisation function."""
    iFair_optimisation.iters += 1
    X_hat, _ = compute_X_hat(X, params, k, alpha=True)

    L_x = compute_reconstruction_loss(X, X_hat)
    L_z = compute_individual_fairness_loss(D_X_f, X_hat)

    criterion = A_x * L_x + A_z * L_z

    if iFair_optimisation.iters % 100 == 0:
        print("step: {}, L_x: {},  L_z: {}, loss:{}\n".format(
                    iFair_optimisation.iters, L_x, L_z, criterion))
        if logs_path != '':
            os.makedirs(logs_path, exist_ok=True)
            with open(os.path.join(logs_path, "logs.txt"), 'a') as f:
                f.write("step: {}, L_x: {},  L_z: {}, loss:{}\n".format(
                    iFair_optimisation.iters, L_x, L_z, criterion))
    return criterion
