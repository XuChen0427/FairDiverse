import os
import numpy as np
from .Abstract_Reranker import Abstract_Reranker
from tqdm import tqdm,trange

r"""
test algorithm
"""

def concave_func(x,t):
    return x**(1-t)

def distance_concave_func(x_anchor, x, t):

    return  (concave_func(x_anchor,t)-concave_func(x,t)) *(1-t) * (x**(-t))


class ConcaveRank(Abstract_Reranker):
    def __init__(self, config, weights = None):
        super().__init__(config, weights)


    def rerank(self, ranking_score, k):
        ## its parameters

        user_size = len(ranking_score)

        t = self.config['t'] #t is from 0 to infty
        assert t>=0
        rerank_list = []

        user_size = len(ranking_score)
        #B_t = user_size * k * self.weights
        B_t = np.ones(self.config['group_num'])
        rerank_list = []

        for u in trange(user_size):
            sort_B_T = np.argsort(B_t)
            anchor_point = sort_B_T[int(self.config['group_num']*self.config['anchor_rate'])]
            norm_B_T = B_t/np.sum(B_t)
            curve_degree = distance_concave_func(norm_B_T[anchor_point], norm_B_T, t)
            #curve_degree = curve_degree/np.sum(curve_degree + 1e-5)
            #print(curve_degree)

            # if u > 3:
            #     exit(0)
            #exit(0)
            rel = ranking_score[u, :]  + np.matmul(self.M, curve_degree)
            result_item = np.argsort(rel)[::-1]
            result_item = result_item[:k]
            rerank_list.append(result_item)
            B_t = B_t + np.sum(self.M[result_item, :], axis=0, keepdims=False)


        return rerank_list
