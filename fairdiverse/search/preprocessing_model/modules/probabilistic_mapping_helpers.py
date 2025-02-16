from __future__ import division
import numpy as np
from scipy.special import softmax
from sklearn.metrics import pairwise


def dist_prototype(X, v, alpha):
    """
    Compute the Euclidean distance between data points and prototypes with a weighted sum.

    :param X : numpy.ndarray
        A 2D array where each row represents a data point.
    :param v : numpy.ndarray
        A 2D array where each row represents a prototype.
    :param alpha : numpy.ndarray
        A 1D array of weights for the features.

    :return : numpy.ndarray
        A 1D array of distances between the data points and prototypes.
    """
    dists = np.sum(alpha * (np.array(X[:, np.newaxis, :]) - np.array(v[np.newaxis, :, :])) ** 2, axis=2)
    return np.sqrt(dists)


def dist_pairwise_group(Xg1, Xg2, alpha):
    """
    Compute the pairwise distances between two sets of data points with feature weights.

    :param Xg1 : numpy.ndarray
        A 2D array representing the first set of data points.
    :param Xg2 : numpy.ndarray
        A 2D array representing the second set of data points.
    :param alpha : numpy.ndarray
        A 1D array of weights for the features.

    :return : numpy.ndarray
        A 2D array containing the pairwise distances between points from Xg1 and Xg2.
    """
    dists = np.sum(alpha * (Xg1[:, np.newaxis, :] - Xg2[np.newaxis, :, :]) ** 2, axis=2)
    return dists


def compute_U_nk(dists):
    """
    Compute the membership matrix based on distances using the softmax function.

    :param dists : numpy.ndarray
        A 2D array of distances between data points and prototypes.

    :return : numpy.ndarray
        A 2D array representing the membership matrix.
    """
    M = softmax(-dists, axis=1)
    return M


def replaceNaNwithMean(X):
    """
    Replaces NaN values in the dataset with the column mean.

    :param X : numpy.ndarray
        A 2D array with potential NaN values.

    :return : numpy.ndarray
        A 2D array with NaN values replaced by the column mean.
    """
    # Obtain mean of columns as you need, nanmean is just convenient.
    col_mean = np.nanmean(X, axis=0)
    # - guarding against a column where all values are NaN
    col_mean[np.isnan(col_mean)] = 0

    # Find indices that need replacement
    inds = np.where(np.isnan(X))

    # Replace NaN values with column means
    X[inds] = np.take(col_mean, inds[1])

    return X


def compute_X_hat(X, params, k, alpha=False):
    """
    Compute the approximated data X_hat using the given parameters.

    :param X : numpy.ndarray
        A 2D array of data points to approximate.
    :param params : numpy.ndarray
        A 1D array containing the model parameters.
    :param k : int
        The number of prototypes.
    :param alpha : bool, optional, default=False
        Whether to use feature-specific weights.

    :return : tuple
        A tuple containing the approximated data X_hat and the membership matrix U_nk.
    """
    N, M = X.shape

    if alpha:
        prototypes = np.matrix(params[M:]).reshape((k, M))
        alpha = params[:M]
    else:
        prototypes = np.matrix(params[k:]).reshape((k, M))
        alpha = np.ones(M)

    distances_prototypes = dist_prototype(X, prototypes, alpha)
    U_nk = compute_U_nk(distances_prototypes)
    X_hat = np.matmul(U_nk, prototypes)

    return np.array(X_hat), U_nk


def get_xhat_y_hat(X, params, k):
    """
    Compute the approximated data X_hat and predicted values Y_hat.

    :param X : numpy.ndarray
        A 2D array of data points to approximate.
    :param params : numpy.ndarray
        A 1D array containing the model parameters.
    :param k : int
        The number of prototypes.

    :return : tuple
        A tuple containing the membership matrix U_nk, the approximated data X_hat,
        and the predicted values Y_hat.
    """
    X_hat, U_nk = compute_X_hat(X, params, k)

    w = params[:k]
    Y_hat = np.clip(
        np.matmul(U_nk, w.reshape((-1, 1))),
        np.finfo(float).eps,
        1.0 - np.finfo(float).eps
    )
    return U_nk, X_hat, Y_hat


def compute_euclidean_distances(X, nonsensitive_column_indices):
    """
    Compute the Euclidean distance matrix based on the non-sensitive columns of the data.

    :param X : numpy.ndarray
        A 2D array of data points.
    :param nonsensitive_column_indices : list
        A list of indices identifying the non-sensitive columns.

    :return : numpy.ndarray
        A 2D array representing the pairwise Euclidean distances between non-sensitive data points.
    """
    X_non_sensitive = X[:, nonsensitive_column_indices]
    if len(X_non_sensitive.shape) > 2:
        X_non_sensitive = X_non_sensitive.squeeze()
    D_X_F = pairwise.euclidean_distances(X_non_sensitive, X_non_sensitive)
    return D_X_F
