import os
import numpy as np
from .Abstract_Reranker import Abstract_Reranker
from tqdm import tqdm,trange

r"""
CPFair

Note that we extend it into multi-group cases and only focus on item-side, slightly different from the original paper

@inproceedings{naghiaei2022cpfair,
  title={Cpfair: Personalized consumer and producer fairness re-ranking for recommender systems},
  author={Naghiaei, Mohammadmehdi and Rahmani, Hossein A and Deldjoo, Yashar},
  booktitle={Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={770--779},
  year={2022}
}
"""

class CPFair(Abstract_Reranker):
    def __init__(self, config, weights = None):
        super().__init__(config, weights)


    def rerank(self, ranking_score, k):
        ## its parameters
        user_size = len(ranking_score)
        B_l = np.zeros(self.group_num)
        lambd = self.config['lambda']
        rerank_list = []

        for u in trange(user_size):
            minimax_reg = lambd * np.matmul(self.M,1-(B_l/(np.sum(B_l)+1e-5)))
            rel = ranking_score[u,:] + minimax_reg
            result_item = np.argsort(rel)[::-1]
            result_item = result_item[:k]
            rerank_list.append(result_item)
            B_l = B_l + np.sum(self.M[result_item,:],axis=0,keepdims=False)


        return rerank_list
