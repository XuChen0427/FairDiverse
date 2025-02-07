from sklearn.metrics import roc_auc_score
import numpy as np

##############ranking metrics #############################

def AUC_score(y_scores, y_true):
    auc_score = roc_auc_score(y_true, y_scores)
    return auc_score

def dcg(scores, k):
    """
    Calculate the Discounted Cumulative Gain (DCG) at rank k.
    """
    scores = np.array(scores)[:k]
    if scores.size:
        return np.sum(scores / np.log2(np.arange(2, scores.size + 2)))
    return 0.0


def NDCG(ranking_list, label_list, k):
    """
    Calculate the Normalized Discounted Cumulative Gain (NDCG) for a ranked list.

    This function computes the NDCG, which is a measure of ranking quality that compares
    the actual ranking of items with an ideal ranking. It is particularly useful for
    evaluating information retrieval systems and recommendation algorithms.

    Parameters
    ----------
    ranking_list : array-like
        A list representing the ranked order of items as predicted by a system.
        The indices correspond to the items' positions in the sorted label list,
        and the values are assumed to be the relevance scores or labels of these items.

    label_list : array-like
        An array of true relevance labels for the items, ordered by their implicit position.
        This list represents the ideal ranking or true relevance of the items.

    k : int
        The cutoff point in the ranked list at which the NDCG is computed. Only the top-k
        elements of `ranking_list` are considered in the calculation.

    Returns
    -------
    float
        The computed NDCG score, which ranges from 0 to 1. A score of 1 indicates a perfect
        ranking where the predicted order matches the ideal relevance order.

    Notes
    -----
    The Discounted Cumulative Gain (DCG) is first computed for both the predicted
    `ranking_list` and the `label_list`, then the NDCG is calculated as the ratio of the
    two DCG values.

    The `dcg` function used within this method should ideally be defined elsewhere and
    computes the DCG given a list of labels and a cutoff `k`.
    """
    #ndcg = 0
    sorted_label = np.sort(label_list)[::-1]
    label_dcg = dcg(sorted_label, k)
    ranking_dcg = dcg(ranking_list, k)
    ndcg = ranking_dcg/label_dcg

    return ndcg

def HR(ranking_list, label_list, k):
    sorted_label = np.sort(label_list)[::-1]
    hr = np.sum(ranking_list[:k])/np.sum(sorted_label[:k])
    return hr


def MRR(ranking_list, k):

    mrr = 0
    for index, i in enumerate(ranking_list[:k]):
        if i > 0:
            mrr = i/(index+1)
            break
    return mrr


##################fairness metric#######################

def reconstruct_utility(utility_list, weights, group_mask):
    if not weights:
        weights = np.ones_like(utility_list)

    utility_list = np.array(utility_list)
    weights = np.array(weights)
    weighted_utility = utility_list * weights

    if group_mask:
        weighted_utility = mask_utility(weighted_utility, group_mask)


    return np.array(weighted_utility)

def mask_utility(utility, group_mask):
    """
    Mask the utility values based on the provided group mask.

    This function filters out the utility values where the corresponding
    group mask element is zero, effectively removing them from the output.

    Parameters
    ----------
    utility : list or numpy.ndarray
        A list or array of utility values to be masked.

    group_mask : list or numpy.ndarray
        A list or array of integers serving as a mask. Each element in
        the utility list is included in the result if the corresponding
        mask element is non-zero.

    Returns
    -------
    numpy.ndarray
        An array of utility values after applying the mask.
    """
    masked_utility = []
    for i, m in enumerate(group_mask):
        if m == 0:
            masked_utility.append(utility[i])

    return np.array(masked_utility)

def MMF(utility_list, ratio=0.5, weights=None, group_mask = None):
    """
    Calculate the Max-min Fairness (MMF) index based on a given utility list.

    Parameters
    - utility_list : array-like
        A list or array representing the utilities of resources or users.
    - ratio : float, optional
        The fraction of the minimum utilities to consider for the MMF calculation. Defaults to 0.5.
    - weights : array-like, optional
        An optional list or array of weights corresponding to each utility in `utility_list`.
        If provided, utilities are multiplied by their respective weights before sorting.
        Defaults to None, implying equal weighting.
    - group_mask : array-like, optional
        An optional list or array used to selectively apply weights. If provided, it must have the same length as
        `utility_list` and `weights`. Defaults to None, indicating no group-based weighting.

    Returns
    - float
        The computed MMF index, representing the fairness of the allocation.
    """
    weighted_utility = reconstruct_utility(utility_list, weights, group_mask)

    MMF_length = int(ratio * len(utility_list))
    utility_sort = np.sort(weighted_utility)

    mmf = np.sum(utility_sort[:MMF_length])/np.sum(weighted_utility)

    return mmf

def MinMaxRatio(utility_list, weights=None, group_mask = None):
    """
    This function computes the minimum-to-maximum ratio of a list of utilities, optionally weighted and grouped.
    Parameters:
    - utility_list (list of float): A list containing numerical utility values.
    - ratio** (float, optional): The default ratio to return if the utility list is empty or invalid. Defaults to 0.5.
    - weights (list of float, optional): A list of weights corresponding to the utilities in utility_list. If provided,
      each utility is multiplied by its respective weight. If None, all utilities are considered with equal weight.
    - group_mask (list of int or bool, optional): A mask indicating groups within the utility_list. If provided, it must
      be of the same length as utility_list. Groups are defined by consecutive True or 1 values. If None, no grouping is applied.

    Returns:
    - float: The computed minimum-to-maximum ratio of the (weighted) utilities. If the utility_list is empty or all utilities
    are zero after weighting and grouping, the function returns the value specified by the `ratio` parameter.
    """
    weighted_utility = reconstruct_utility(utility_list, weights, group_mask)
    return np.min(weighted_utility) / np.max(weighted_utility)


def Gini(utility_list, weights=None, group_mask = None):
    """
    This function computes the Gini coefficient, a measure of statistical dispersion intended to represent income inequality within a nation or social group.
    The Gini coefficient is calculated based on the cumulative distribution of values in `utility_list`, which can optionally be weighted and masked.

    Parameters
    - utility_list: array_like
        A 1D array representing individual utilities. The utilities are used to compute the Gini coefficient.
    - weights: array_like, optional
        A 1D array of weights corresponding to `utility_list`. If provided, each utility value is multiplied by its respective weight before calculating the Gini coefficient. Defaults to None, implying equal weighting.
    - group_mask: array_like, optional
        A 1D boolean array used to selectively include elements from `utility_list`. If provided, only the elements where the mask is True are considered in the calculation. Defaults to None, meaning all elements are included.

    Returns
    - float
        The computed Gini coefficient, ranging from 0 (perfect equality) to 1 (maximal inequality).
    """
    weighted_utility = reconstruct_utility(utility_list, weights, group_mask)

    values = np.sort(weighted_utility)
    n = len(values)

    # gini compute
    cumulative_sum = np.cumsum(values)
    gini = (n + 1 - 2 * (np.sum(cumulative_sum) / cumulative_sum[-1])) / n

    return gini

def Entropy(utility_list, weights=None, group_mask = None):
    """
    Calculate the entropy of a distribution given by `utility_list`, optionally
    weighted by `weights` and filtered by `group_mask`. Entropy measures the
    disorder or uncertainty in the distribution.

    Parameters
    - utility_list : list or array-like
        A list or array representing utility values for each item.
    - weights : list or array-like, optional
        A list or array of weights corresponding to each utility value. If not provided,
        all utilities are considered equally weighted. Defaults to None.
    - group_mask : list or array-like, optional
        A boolean mask indicating which utilities to include in the calculation.
        If not provided, all utilities are included. Defaults to None.

    Returns
    - float
        The calculated entropy of the (potentially weighted and masked) distribution.

    Notes
    - Entropy is calculated as H = -sum(p * log2(p)), where p is the probability of each event.
    - Probabilities are normalized to ensure their sum equals 1.
    - To avoid taking the log of zero, a small constant (1e-9) is added to each probability before calculating the entropy.
    """
    weighted_utility = reconstruct_utility(utility_list, weights, group_mask)

    values = np.array(weighted_utility)
    values = values / np.sum(values)  # 归一化

    # H = - sum(p * log2(p))
    # avoid 0 case
    entropy_value = -np.sum(values * np.log2(values + 1e-9))
    return entropy_value

def ElasticFair(utils,t):

    if t != 0:
        sign = np.sign(1-t)
        utils_g = np.sum(np.power(utils,1-t))
        utils_g = np.power(utils_g, 1/t)
        return sign * utils_g
    else:
        entropy = - np.sum(utils * np.log(utils))
        return np.exp(entropy)

def EF(utility_list, M = 50, weights=None, group_mask = None):
    utility_list = reconstruct_utility(utility_list, weights, group_mask)
    utility_list = utility_list / np.sum(utility_list)

    t = np.linspace(1 - M, 1 + M, 200)
    fair = []
    for i in t:
        fair.append(ElasticFair(utility_list, i))
    integral = np.trapz(fair, t)
    return integral/(2*M*len(utility_list))
