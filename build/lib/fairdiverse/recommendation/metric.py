from sklearn.metrics import roc_auc_score
import numpy as np

##############ranking metrics #############################

def AUC_score(y_scores, y_true):
    r"""AUC_ (also known as Area Under Curve) is used to evaluate the two-class model, referring to
        the area under the ROC curve.

        .. _AUC: https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve

        Note:
            This metric does not calculate group-based AUC which considers the AUC scores
            averaged across users. It is also not limited to k. Instead, it calculates the
            scores on the entire prediction results regardless the users. We call the interface
            in `scikit-learn`, and code calculates the metric using the variation of following formula.

        .. math::
            \mathrm {AUC} = \frac {{{M} \times {(N+1)} - \frac{M \times (M+1)}{2}} -
            \sum\limits_{i=1}^{M} rank_{i}} {{M} \times {(N - M)}}

        :math:`M` denotes the number of positive items.
        :math:`N` denotes the total number of user-item interactions.
        :math:`rank_i` denotes the descending rank of the i-th positive item.
    """
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
    r"""NDCG_ (also known as normalized discounted cumulative gain) is a measure of ranking quality,
    where positions are discounted logarithmically. It accounts for the position of the hit by assigning
    higher scores to hits at top ranks.

    .. _NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG

    .. math::
        \mathrm {NDCG@K} = \frac{1}{|U|}\sum_{u \in U} (\frac{1}{\sum_{i=1}^{\min (|R(u)|, K)}
        \frac{1}{\log _{2}(i+1)}} \sum_{i=1}^{K} \delta(i \in R(u)) \frac{1}{\log _{2}(i+1)})

    :math:`\delta(·)` is an indicator function.
    """
    sorted_label = np.sort(label_list)[::-1]
    label_dcg = dcg(sorted_label, k)
    ranking_dcg = dcg(ranking_list, k)
    ndcg = ranking_dcg/label_dcg

    return ndcg

def HR(ranking_list, label_list, k):
    r"""HR_ (also known as truncated Hit-Ratio) is a way of calculating how many 'hits'
        you have in an n-sized list of ranked items. If there is at least one item that falls in the ground-truth set,
        we call it a hit.

        .. _HR: https://medium.com/@rishabhbhatia315/recommendation-system-evaluation-metrics-3f6739288870

        .. math::
            \mathrm {HR@K} = \frac{1}{|U|}\sum_{u \in U} \delta(\hat{R}(u) \cap R(u) \neq \emptyset),

        :math:`\delta(·)` is an indicator function. :math:`\delta(b)` = 1 if :math:`b` is true and 0 otherwise.
        :math:`\emptyset` denotes the empty set.
    """
    sorted_label = np.sort(label_list)[::-1]
    hr = np.sum(ranking_list[:k])/np.sum(sorted_label[:k])
    return hr


def MRR(ranking_list, k):
    r"""The MRR_ (also known as Mean Reciprocal Rank) computes the reciprocal rank
        of the first relevant item found by an algorithm.

        .. _MRR: https://en.wikipedia.org/wiki/Mean_reciprocal_rank

        .. math::
           \mathrm {MRR@K} = \frac{1}{|U|}\sum_{u \in U} \frac{1}{\operatorname{rank}_{u}^{*}}

        :math:`{rank}_{u}^{*}` is the rank position of the first relevant item found by an algorithm for a user :math:`u`.
    """
    mrr = 0
    for index, i in enumerate(ranking_list[:k]):
        if i > 0:
            mrr = i/(index+1)
            break
    return mrr


##################fairness metric#######################

def reconstruct_utility(utility_list, weights, group_mask):
    """
        Reconstruct utility by re-weighting them and masking the utility of certain unused groups.
        :param utility_list: array for item/user utilities
        :param weights: array for item/user utilities weights
        :param group_mask: bool array for whether computed the group utilityes
        :return: re-constructed utility array
    """

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


    :param utility: array for item/user utilities
    :param group_mask: bool array for whether computed the group utilityes
    :return: masked utility array
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
    :param utility: array-like
        A list or array representing the utilities of resources or users.
    :param ratio: float, optional
        The fraction of the minimum utilities to consider for the MMF calculation. Defaults to 0.5.
    :param ratio: float, optional
        The fraction of the minimum utilities to consider for the MMF calculation. Defaults to 0.5.
    :param ratio: float, optional
        The fraction of the minimum utilities to consider for the MMF calculation. Defaults to 0.5.
    :param weights : array-like, optional
        An optional list or array of weights corresponding to each utility in `utility_list`.
        If provided, utilities are multiplied by their respective weights before sorting.
        Defaults to None, implying equal weighting.
    :param group_mask : array-like, optional
        An optional list or array used to selectively apply weights. If provided, it must have the same length as
        `utility_list` and `weights`. Defaults to None, indicating no group-based weighting.

    :return: The computed MMF index, representing the fairness of the allocation.
    """
    weighted_utility = reconstruct_utility(utility_list, weights, group_mask)

    MMF_length = int(ratio * len(utility_list))
    utility_sort = np.sort(weighted_utility)

    mmf = np.sum(utility_sort[:MMF_length])/np.sum(weighted_utility)

    return mmf

def MinMaxRatio(utility_list, weights=None, group_mask = None):
    """
    This function computes the minimum-to-maximum ratio of a list of utilities, optionally weighted and grouped.

    :param utility_list (list of float): A list containing numerical utility values.
    :param weights (list of float, optional): A list of weights corresponding to the utilities in utility_list. If provided,
      each utility is multiplied by its respective weight. If None, all utilities are considered with equal weight.
    :param group_mask (list of int or bool, optional): A mask indicating groups within the utility_list. If provided, it must
      be of the same length as utility_list. Groups are defined by consecutive True or 1 values. If None, no grouping is applied.

    :return: float: The computed minimum-to-maximum ratio of the (weighted) utilities.
    """
    weighted_utility = reconstruct_utility(utility_list, weights, group_mask)
    return np.min(weighted_utility) / np.max(weighted_utility)


def Gini(utility_list, weights=None, group_mask = None):
    """
    This function computes the Gini coefficient, a measure of statistical dispersion intended to represent income inequality within a nation or social group.
    The Gini coefficient is calculated based on the cumulative distribution of values in `utility_list`, which can optionally be weighted and masked.

    :param utility_list: array_like
        A 1D array representing individual utilities. The utilities are used to compute the Gini coefficient.
    :param weights: array_like, optional
        A 1D array of weights corresponding to `utility_list`. If provided, each utility value is multiplied by its respective weight before calculating the Gini coefficient. Defaults to None, implying equal weighting.
    :param group_mask: array_like, optional
        A 1D boolean array used to selectively include elements from `utility_list`. If provided, only the elements where the mask is True are considered in the calculation. Defaults to None, meaning all elements are included.

    :return: float: The computed Gini coefficient, ranging from 0 (perfect equality) to 1 (maximal inequality).
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

    :param utility_list: list or array-like
        A list or array representing utility values for each item.
    :param weights: list or array-like, optional
        A list or array of weights corresponding to each utility value. If not provided,
        all utilities are considered equally weighted. Defaults to None.
    :param group_mask : list or array-like, optional
        A boolean mask indicating which utilities to include in the calculation.
        If not provided, all utilities are included. Defaults to None.

    :return: float: The calculated entropy of the (potentially weighted and masked) distribution.

    Notes
    - Entropy is calculated as H = -sum(p * log2(p)), where p is the probability of each event.
    - Probabilities are normalized to ensure their sum equals 1.
    - To avoid taking the log of zero, a small constant (1e-9) is added to each probability before calculating the entropy.
    """
    weighted_utility = reconstruct_utility(utility_list, weights, group_mask)

    values = np.array(weighted_utility)
    values = values / np.sum(values)

    # H = - sum(p * log2(p))
    # avoid 0 case
    entropy_value = -np.sum(values * np.log2(values + 1e-9))
    return entropy_value



