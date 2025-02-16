import os
import numpy as np
from .Abstract_Reranker import Abstract_Reranker
from tqdm import tqdm,trange
import cvxpy as cp
import math

r"""
P-MMF: an online algorithm to support the worst-off groups. P-MMF is awarded as WWW spotlight best-paper-candidate

@inproceedings{xu2023p,
  title={P-MMF: Provider max-min fairness re-ranking in recommender system},
  author={Xu, Chen and Chen, Sirui and Xu, Jun and Shen, Weiran and Zhang, Xiao and Wang, Gang and Dong, Zhenhua},
  booktitle={Proceedings of the ACM Web Conference 2023},
  pages={3701--3711},
  year={2023}
}

"""

def compute_projection_maxmin_fairness_with_order(ordered_tilde_dual, rho, lambd):

    m = len(rho)
    answer = cp.Variable(m)
    objective = cp.Minimize(cp.sum_squares(cp.multiply(rho,answer) - cp.multiply(rho, ordered_tilde_dual)))
    #objective = cp.Minimize(cp.sum(cp.multiply(rho,answer) - cp.multiply(rho, ordered_tilde_dual)))
    constraints = []
    for i in range(1, m+1):
        constraints += [cp.sum(cp.multiply(rho[:i],answer[:i])) >= -lambd]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    #print(type(result))
    #exit(0)
    #print(type(answer.value))
    return answer.value



def compute_next_dual(eta, rho, dual, gradient, lambd):
    #rho = self.problem.data.rho
    tilde_dual = dual - eta*gradient/rho/rho
    order = np.argsort(tilde_dual*rho)
    ordered_tilde_dual = tilde_dual[order]
    # print ordered_tilde_dual*rho
    ordered_next_dual = compute_projection_maxmin_fairness_with_order(ordered_tilde_dual, rho[order], lambd)
    # print(ordered_next_dual)
    # print("tilde_dual", rho*tilde_dual)
    # print("next_dual", rho*ordered_next_dual[order.argsort()])
    return ordered_next_dual[order.argsort()]

class PMMF(Abstract_Reranker):
    def __init__(self, config, weights = None):
        super().__init__(config, weights)


    def rerank(self, ranking_score, k):
        ## its parameters

        lambd = self.config['lambda']
        learning_rate = self.config['learning_rate']
        gamma = self.config['gamma']


        user_size = len(ranking_score)
        assert len(self.weights) == self.config['group_num']
        B_t = user_size * k * self.weights

        #B_l = np.zeros(self.group_num)
        rerank_list = []

        mu_t = np.zeros(self.config['group_num'])
        eta = learning_rate / math.sqrt(self.config['item_num'])
        gradient_cusum = np.zeros(self.config['group_num'])

        for u in trange(user_size):
            x_title = ranking_score[u, :] - np.matmul(self.M, mu_t)
            mask = np.matmul(self.M, (B_t > 0).astype(np.float32))
            mask = (1.0 - mask) * -10000.0
            x = np.argsort(x_title + mask, axis=-1)[::-1]
            x_allocation = x[:k]
            re_allocation = np.argsort(ranking_score[u, x_allocation])[::-1]
            x_allocation = x_allocation[re_allocation]
            rerank_list.append(x_allocation)
            B_t = B_t - np.sum(self.M[x_allocation], axis=0, keepdims=False)
            gradient = -np.mean(self.M[x_allocation], axis=0, keepdims=False) + self.weights

            # gradient_list.append(gradient)
            gradient = gamma * gradient + (1 - gamma) * gradient_cusum
            gradient_cusum = gradient
            # gradient = -(B_0-B_t)/((t+1)*K) + rho
            for g in range(1):
                mu_t = compute_next_dual(eta, self.weights, mu_t, gradient, lambd)

        return rerank_list

