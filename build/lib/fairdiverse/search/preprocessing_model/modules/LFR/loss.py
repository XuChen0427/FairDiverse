
"""Learning fair representations is a pre-processing technique that finds a
    latent representation which encodes the data well but obfuscates information
    about protected attributes [2]_.
    References:
        .. [2] R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork,  "Learning
           Fair Representations." International Conference on Machine Learning,
           2013.
    Based on code from https://github.com/zjelveh/learning-fair-representations
    """

import numpy as np
import os
from ..probabilistic_mapping_helpers import get_xhat_y_hat
def compute_reconstruction_loss(x_orig, x_hat):
    """Computes the reconstruction loss (L_x)."""

    return np.mean((x_hat - x_orig) * (x_hat - x_orig))

def compute_LFR_utility_loss(y_true, y_hat):
    """Computes the utility loss (L_y) using binary cross-entropy between positive and negative samples."""
    return -np.mean(y_true * np.log(y_hat) + (1. - y_true) * np.log(1. - y_hat))


def compute_LFR_fairness_loss_pairwise(M_unpriv, M_priv):
    """Computes the fairness loss (L_z)."""
    group_gap = np.mean(abs(np.mean(M_unpriv, axis=0) - np.mean(M_priv, axis=0)))

    return group_gap
def compute_LFR_group_fairness_loss(X, params, k, group_weights, sensitive_groups, sensitive_indexes, biggest_gap):
    """LFR optimisation function"""
    L_z = 0
    for s_attr in sensitive_groups["privileged_groups"].keys():
        privileged_groups = sensitive_groups["privileged_groups"][s_attr]
        unprivileged_groups = sensitive_groups["unprivileged_groups"][s_attr]
        for index_advantaged_group in range(len(privileged_groups)):
            for s_index in sensitive_indexes:
                mask_privileged = np.isin(element=X[:, s_index],
                                          test_elements=privileged_groups[index_advantaged_group]).reshape(-1)
                if sum(mask_privileged) != 0:
                    for index_unprivileged_group in range(index_advantaged_group + 1, len(unprivileged_groups)):
                        if unprivileged_groups[index_unprivileged_group] != privileged_groups[index_advantaged_group]:
                            mask_unprivileged = np.isin(element=X[:, s_index],
                                             test_elements=unprivileged_groups[index_unprivileged_group]).reshape(-1)
                            if sum(mask_unprivileged) != 0:
                                if unprivileged_groups[index_unprivileged_group] != privileged_groups[
                                    index_advantaged_group]:
                                    weight = abs(
                                        group_weights[privileged_groups[index_advantaged_group]] - group_weights[
                                            unprivileged_groups[index_unprivileged_group]]) + 1

                                    M_unpriv, X_hat, Y_hat = get_xhat_y_hat(X[mask_unprivileged], params, k)
                                    M_priv, X_hat, Y_hat = get_xhat_y_hat(X[mask_privileged], params, k)
                                    group_gap = compute_LFR_fairness_loss_pairwise(M_unpriv, M_priv)
                                    if biggest_gap:
                                        if biggest_gap_groups < group_gap:
                                            biggest_gap_groups = group_gap
                                            L_z = weight * group_gap
                                    else:
                                        L_z += weight * group_gap

    return L_z


def LFR_optimisation(
        params, X, Y, sensitive_groups, sensitive_indexes, k=10,
        A_x=0.01, A_y=0.1, A_z=0.5,
        group_weights=None, biggest_gap=False, logs_path=''):
    """
    Objective function for the Learned Fair Representations (LFR) optimization.
    """

    M, X_hat, Y_hat = get_xhat_y_hat(X, params, k)

    L_x = compute_reconstruction_loss(X, X_hat)
    L_y = compute_LFR_utility_loss(Y, Y_hat)
    L_z = compute_LFR_group_fairness_loss(X, params, k, group_weights, sensitive_groups, sensitive_indexes,
                                          biggest_gap)

    # Compute total loss
    total_loss = A_x * L_x + A_y * L_y + A_z * L_z

    # Logging
    if LFR_optimisation.steps % 100 == 0:
        log_msg = f"step: {LFR_optimisation.steps}, loss: {total_loss}, " \
                  f"L_x: {L_x}, L_y: {L_y}, L_z: {L_z}"
        print(log_msg)

        if logs_path:
            os.makedirs(logs_path, exist_ok=True)
            with open(os.path.join(logs_path, "logs.txt"), 'a') as f:
                f.write(log_msg + "\n")

    LFR_optimisation.steps += 1
    return total_loss
