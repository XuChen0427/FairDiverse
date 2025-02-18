import os
import numpy as np
from .Abstract_Reranker import Abstract_Reranker
from tqdm import tqdm,trange

r"""
min-regularizer, a heuristic baseline aim to support the worst-off providers. It is proposed by

@inproceedings{xu2023p,
  title={P-MMF: Provider max-min fairness re-ranking in recommender system},
  author={Xu, Chen and Chen, Sirui and Xu, Jun and Shen, Weiran and Zhang, Xiao and Wang, Gang and Dong, Zhenhua},
  booktitle={Proceedings of the ACM Web Conference 2023},
  pages={3701--3711},
  year={2023}
}

"""

class min_regularizer(Abstract_Reranker):
    def __init__(self, config, weights = None):
        super().__init__(config, weights)


    def rerank(self, ranking_score, k):
        ## its parameters
        user_size = len(ranking_score)
        B_l = np.zeros(self.group_num)
        lambd = self.config['lambda']
        rerank_list = []

        for u in trange(user_size):
            minimax_reg = lambd * np.matmul(self.M,(-B_l + np.min(B_l))/(self.weights * user_size))
            rel = ranking_score[u,:] + minimax_reg
            result_item = np.argsort(rel)[::-1]
            result_item = result_item[:k]
            rerank_list.append(result_item)
            B_l = B_l + np.sum(self.M[result_item,:],axis=0,keepdims=False)

        return rerank_list
