import os
import numpy as np
from .Abstract_Reranker import Abstract_Reranker
from tqdm import tqdm,trange
import copy
r"""
Welf

Note that since the author do not publish the codes, we implement according to the equations of the paper, could be slightly different with the results
Welf is a item-side fairness algorithm, it may not perform good in multi-group fairness settings.

@article{do2021two,
  title={Two-sided fairness in rankings via Lorenz dominance},
  author={Do, Virginie and Corbett-Davies, Sam and Atif, Jamal and Usunier, Nicolas},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={8596--8608},
  year={2021}
}
"""

def phi_func(x,alpha):
    if x >= 0:
        return np.power(x+1e-8,alpha)
    else:
        raise ValueError("x:",x)

class Welf(Abstract_Reranker):
    def __init__(self, config, weights = None):
        super().__init__(config, weights)


    def rerank(self, ranking_score, k):
        ## its parameters
        lambd = self.config['lambda']
        alpha1 = self.config['alpha1']
        alpha2 = self.config['alpha2']
        iter_num = self.config['iter_num']

        user_size = len(ranking_score)
        P_t = copy.copy(ranking_score)
        for iter in trange(iter_num):
            gamma = 2/(iter + 2)
            title_P = np.zeros((user_size, self.config['item_num']))
            B_l = np.zeros(self.config['group_num'])

            for t in range(user_size):
                #title_P = np.zeros((item_num,K))

                recommended_eatch_iter = np.argsort(P_t[t,:],axis=-1)[::-1]
                recommended_eatch_iter = recommended_eatch_iter[:k]

                B_l = B_l + np.sum(self.M[recommended_eatch_iter,:],axis=0,keepdims=False)
                title_P[t,:] = alpha1 * ranking_score[t,:] * phi_func(np.sum(ranking_score[t,:]), alpha1-1)
            #print(B_l.astype(np.int))
            for i in range(self.item_num):
                title_P[:,i] += lambd * alpha2 * np.ones(user_size) * phi_func(np.sum(self.M[i,:]*(1-B_l/(np.sum(B_l)+1e-5))), alpha2-1)

            P_t = (1-gamma) * P_t + gamma * title_P

        rerank_list = []
        for t in range(user_size):
            items = np.argsort(P_t[t,:],axis=-1)[::-1]
            x_allocation = items[:k]
            re_allocation = np.argsort(ranking_score[t,x_allocation])[::-1]
            x_allocation = x_allocation[re_allocation]
            rerank_list.append(x_allocation)

        return rerank_list
