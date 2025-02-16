import os
import numpy as np
from .Abstract_Reranker import Abstract_Reranker
from tqdm import tqdm,trange
import ot
import cvxpy as cp

r"""
TaxRank, an algorithm to transfer the item-side fair re-ranking phase as a taxation process.
The TaxRank is designed for item-side fairness, its performance may be not good in group-side

TaxRank is the probability method, in original paper, it takes the expectation as the results, however, here we only takes one turn for re-rank evaluation
For the item-side probability evaluation, please ref: https://github.com/XuChen0427/Tax-rank


@inproceedings{xu2024taxation,
  title={A Taxation Perspective for Fair Re-ranking},
  author={Xu, Chen and Ye, Xiaopeng and Wang, Wenjie and Pang, Liang and Xu, Jun and Chua, Tat-Seng},
  booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={1494--1503},
  year={2024}
}



"""

class TaxRank(Abstract_Reranker):
    def __init__(self, config, weights = None):
        super().__init__(config, weights)


    def rerank(self, ranking_score, k):
        ## its parameters

        user_size = len(ranking_score)

        t = self.config['t'] #t is from 0 to infty
        assert t>=0
        rerank_list = []

        e = cp.Variable(self.config['item_num'])
        con = [e >= 0, e <= user_size, cp.sum(e) == k * user_size]

        #sorted_ranking_score = np.sort(ranking_score)[::-1]
        eta = k * np.sum(ranking_score, axis=0).reshape(self.config['item_num'])
        item_weight = np.matmul(self.M, self.weights)

        if t == 1:
            fairness = cp.sum(item_weight * cp.multiply(eta, cp.log(e)))
        else:

            fairness = cp.sum(item_weight * cp.multiply(eta, cp.power(e, 1 - t) / (1 - t)))

        obj = cp.Maximize(fairness)

        prob = cp.Problem(obj, con)
        prob.solve()


        e_value = e.value

        group_per_item = np.sum(self.M, axis=0)
        #print(group_per_item)
        item2group_weight = np.matmul(self.M, group_per_item)
        e_value = e_value/item2group_weight
        #print(e_value)
        #exit(0)
        e_value = (e_value/np.sum(e_value)) * k * user_size




        ###start to conduct OT algorithm
        user_size = len(ranking_score)

        Ks = [k for i in range(user_size)]
        answer = ot.sinkhorn(Ks, e_value, 1.0-ranking_score, self.config['ot_lambda'])

        answer = np.where(np.isnan(answer), 0, answer)
        item_prob = np.clip(answer, 0, 1)

        for u in range(user_size):
            mask = np.argsort(ranking_score[u])[:-self.config['sample_num']]
            item_prob[u][mask] = 0.0
            norm_prob = item_prob[u]/np.sum(item_prob[u])

            #print(norm_prob)

            ##  it is the probality method, in original paper, it takes the expectation as the results, however, here we only takes one turn for re-rank
            rerank_items = np.random.choice(self.config['item_num'], size=k, replace=False, p=norm_prob)
            rerank_list.append(rerank_items)

        #recommend_list = answer

        return rerank_list
