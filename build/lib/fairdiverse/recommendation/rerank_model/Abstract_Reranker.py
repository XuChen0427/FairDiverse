import os
import numpy as np
import json
from scipy.sparse import coo_matrix, csr_matrix
from ..utils import Build_Adjecent_Matrix


class Abstract_Reranker(object):
    def __init__(self, config, weights = None):
        self.config = config
        self.item_num = config['item_num']
        self.group_num = config['group_num']
        if not weights:
            weights = np.ones(self.group_num)
        self.weights = weights
        self.M, self.iid2pid = Build_Adjecent_Matrix(config)


    def rerank(self, ranking_score, k):
        """
           Re-ranks the items based on the initial ranking scores and a fairness regularization term.

           This function performs re-ranking of items for each user by incorporating a fairness regularization term
           (`minimax_reg`) that adjusts the ranking scores to promote fairness across groups. The re-ranked list of
           items is returned for each user.

           :param ranking_score: A 2D array (or tensor) of ranking scores for all items, with shape (user_size, item_num).
                                  Each row corresponds to the scores for a user and each column corresponds to an item.
           :param k: The number of top-ranked items to return for each user.
           :return: A list of re-ranked item indices for each user, with the top `k` items for each user.
        """
        pass
