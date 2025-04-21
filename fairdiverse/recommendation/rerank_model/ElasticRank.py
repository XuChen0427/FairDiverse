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


class ElasticRank(Abstract_Reranker):
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
        V_t = np.ones(self.config['group_num'])

        rerank_list = []


        curve_degrees = []
        v_t = []

        for u in trange(user_size):
            sort_B_T = np.argsort(V_t)
            anchor_point = sort_B_T[int(self.config['group_num']*self.config['anchor_rate'])]
            norm_B_T = V_t/np.sum(V_t)
            curve_degree = distance_concave_func(norm_B_T[anchor_point], norm_B_T, t)
            curve_degrees.append(curve_degree)
            rel = ranking_score[u, :]  + self.config['lambda'] * np.matmul(self.M, curve_degree)
            result_item = np.argsort(rel)[::-1]
            result_item = result_item[:k]
            scores = ranking_score[u, result_item] #[K]
            rerank_list.append(result_item)
            B_t = B_t + np.sum(self.M[result_item, :], axis=0, keepdims=False)
            V_t = V_t + np.matmul(scores, self.M[result_item, :])
            #print(V_t)
            v_t.append(V_t)

            #exit(0)
        # v_t = np.array(v_t)
        # curve_degrees = np.array(curve_degrees)
        # distance_dir = "C:\lab\pairwise-fairness\exp\distance_exp"
        # np.save(os.path.join(distance_dir,"utilities.npy"), v_t)
        # np.save(os.path.join(distance_dir, "curve_degree.npy"), curve_degrees)
        # exit(0)


        return rerank_list
