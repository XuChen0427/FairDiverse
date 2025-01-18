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

    masked_utility = []
    for i, m in enumerate(group_mask):
        if m == 0:
            masked_utility.append(utility[i])

    return np.array(masked_utility)

def MMF(utility_list, ratio=0.5, weights=None, group_mask = None):
    weighted_utility = reconstruct_utility(utility_list, weights, group_mask)

    MMF_length = int(ratio * len(utility_list))
    utility_sort = np.sort(weighted_utility)

    mmf = np.sum(utility_sort[:MMF_length])/np.sum(weighted_utility)

    return mmf

def MinMaxRatio(utility_list, ratio=0.5, weights=None, group_mask = None):
    weighted_utility = reconstruct_utility(utility_list, weights, group_mask)
    return np.min(weighted_utility) / np.max(weighted_utility)


def Gini(utility_list, weights=None, group_mask = None):
    weighted_utility = reconstruct_utility(utility_list, weights, group_mask)

    values = np.sort(weighted_utility)
    n = len(values)

    # gini compute
    cumulative_sum = np.cumsum(values)
    gini = (n + 1 - 2 * (np.sum(cumulative_sum) / cumulative_sum[-1])) / n

    return gini

def Entropy(utility_list, weights=None, group_mask = None):
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
