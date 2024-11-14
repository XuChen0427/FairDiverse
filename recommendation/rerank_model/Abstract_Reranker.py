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
        pass

    def full_predict(self, user, items):
        pass