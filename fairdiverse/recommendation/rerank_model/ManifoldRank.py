import os
import numpy as np
from .Abstract_Reranker import Abstract_Reranker
from tqdm import tqdm,trange
from scipy.stats import entropy, skew

r"""
ManifoldRank: it is a method for re-ranking, which interpret reranking as manifold optmization to find best equilibrium.
"""

class ManifoldRank(Abstract_Reranker):
    def __init__(self, config, weights = None):
        super().__init__(config, weights)

    def predict_learning_rate(self, scores):
        v = scores[scores > 1e-2]
        prob = np.exp(v - v.max())
        prob /= prob.sum()
        E = entropy(prob, base=1.2)
        S = skew(v)
        return np.clip(self.config['a_e'] * (E - self.config['b']) - self.config['a_s'] * S, 0.08, 10)

    def rerank(self, ranking_score, k):
        user_size = len(ranking_score)
        B_l = np.ones(self.group_num)
        beta = self.config['beta']
        alpha = self.config['alpha']
        rerank_list = []
        V_t = np.ones(self.config['group_num'])
        c = np.sign(alpha*beta)

        for u in trange(user_size):
            G = np.sum(np.power(V_t, beta))
            r_x = np.power(G,alpha-1) * np.power(V_t, beta)
            lr = self.predict_learning_rate(ranking_score[u,:])
            f_score = r_x * beta * (1/V_t+(alpha-1)*np.power(V_t, beta-1)/G)
            f_score = f_score/np.sum(f_score)
            signal = np.sign(alpha*beta * ((alpha - 1) * beta * np.power(V_t, beta) + (beta - 1) * G))
            f_score = f_score * signal
            f_score = np.matmul(self.M, f_score)
            rel = ranking_score[u,:] - c * lr * f_score
            result_item = np.argsort(rel)[::-1]
            result_item = result_item[:k]
            scores = ranking_score[u, result_item]
            rerank_list.append(result_item)
            B_l = B_l + np.sum(self.M[result_item,:],axis=0,keepdims=False)
            V_t = V_t + np.matmul(scores, self.M[result_item, :])

        return rerank_list
