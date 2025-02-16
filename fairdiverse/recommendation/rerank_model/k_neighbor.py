import os
import numpy as np
from .Abstract_Reranker import Abstract_Reranker
from tqdm import tqdm,trange

r"""
K-neighbor: it is a heuristic baseline for re-ranking: every user arrives, it will only choose the least-k worst-off group's items to recommend
"""

class k_neighbor(Abstract_Reranker):
    def __init__(self, config, weights = None):
        super().__init__(config, weights)


    def rerank(self, ranking_score, k):
        ## its parameters
        user_size = len(ranking_score)
        B_l = np.zeros(self.group_num)
        rerank_list = []

        for u in trange(user_size):
            worstoff_groups = np.argsort(B_l)
            worstoff_groups = worstoff_groups[:self.config['nearest_k']]
            target_onehot_group = np.zeros(self.config['group_num'])
            target_onehot_group[worstoff_groups] = 1
            target_items = np.matmul(self.M, target_onehot_group)
            recommended_mask = (1 - target_items) * -10000.0
            rel = ranking_score[u,:]
            result_item = np.argsort(rel+recommended_mask)[::-1]
            result_item = result_item[:k]
            rerank_list.append(result_item)
            B_l = B_l + np.sum(self.M[result_item,:],axis=0,keepdims=False)


        return rerank_list
