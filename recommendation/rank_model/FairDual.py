import numpy as np
from .Abstract_Ranker import Abstract_Reweigher
import cvxpy as cp
import time

r"""
FairDual
################################################
"""




class FairDual(Abstract_Reweigher):
    def __init__(self, config, group_weight):
        super().__init__(config)
        self.group_weight = group_weight

    def reset_parameters(self, train_len):
        self.train_len = train_len
        self.update_len = self.config['update_epoch'] * train_len
        self.C_t = self.config['s_k'] * self.update_len * self.group_weight
        self.mu = np.ones(self.config['group_num'])
        self.gradient_cusum = 0

        self.exposure_count = 0


    def reweight(self, input_dict):
        ###here the items are sampled items and predict the topk
        items = input_dict['sample_items']
        batch_size = len(items)

        B_t = np.sum(self.M[items], axis=1, keepdims=False)
        batch_weight = np.mean(B_t * self.mu, axis=-1, keepdims=False)
        batch_weight = 1.0 - batch_weight / np.sum(batch_weight)

        #############update parameters
        D_t = np.sum(B_t, axis=0, keepdims=False)
        self.C_t = self.C_t - D_t
        gradient = -D_t / (batch_size * self.config['s_k']) + self.C_t / (
                    self.train_len * self.config['s_k'])
        gradient = self.config['alpha'] * gradient + (
                    1 - self.config['alpha']) * self.gradient_cusum
        self.gradient_cusum = gradient

        for g in range(1):

            self.mu = self.compute_next_dual(self.config['eta'], self.group_weight, self.mu, gradient,
                                        self.config['lambd'])

        Dual_weight = batch_weight
        #print(self.mu)
        #batch_weight = np.exp(batch_weight) / np.sum(np.exp(batch_weight))

        # items = input_dict['items']
        #
        # adj_matrix = self.M[items]
        #
        # B_t = np.sum(adj_matrix, axis=0, keepdims=False)
        # self.exposure_count = self.exposure_count + B_t
        # norm_count = self.group_weight * self.exposure_count / np.sum(self.exposure_count)
        # batch_weight = np.matmul(adj_matrix, norm_count)
        # batch_weight = batch_weight / np.sum(batch_weight)
        # IPS_weight = 1/(batch_weight+0.1)


        return Dual_weight



    def compute_next_dual(self, eta, rho, dual, gradient, lambd):
        tilde_dual = dual - eta * gradient / rho / rho
        order = np.argsort(tilde_dual * rho)
        ordered_tilde_dual = tilde_dual[order]
        ordered_next_dual = self.map_layer(ordered_tilde_dual, rho[order], lambd)
        return ordered_next_dual[order.argsort()]

    def map_layer(self, ordered_tilde_dual, rho, lambd):
        m = len(rho)
        answer = cp.Variable(m)
        objective = cp.Minimize(cp.sum_squares(cp.multiply(rho, answer) - cp.multiply(rho, ordered_tilde_dual)))
        # objective = cp.Minimize(cp.sum(cp.multiply(rho,answer) - cp.multiply(rho, ordered_tilde_dual)))
        constraints = []
        for i in range(1, m + 1):
            constraints += [cp.sum(cp.multiply(rho[:i], answer[:i])) >= -lambd]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        return answer.value

