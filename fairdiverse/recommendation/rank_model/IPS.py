import numpy as np
from .Abstract_Ranker import Abstract_Reweigher

r"""
Inverse Propensity Scores Re-weighter
################################################

The heuristic baselines, which utilizes the item popularity weight as the inverse propensity scores for the different samples
Item-side fairness of large language model-based recommendation system, WWW Web4Good 2024, Jiang et al.

"""

class IPS(Abstract_Reweigher):
    def __init__(self, config, group_weight):
        super().__init__(config)
        self.variance_control = config['M']
        self.group_weight = group_weight


    def reweight(self, input_dict):
        items = input_dict['items']

        adj_matrix = self.M[items]

        B_t = np.sum(adj_matrix, axis=0, keepdims=False)
        self.exposure_count = self.exposure_count + B_t
        norm_count = self.group_weight * self.exposure_count / np.sum(self.exposure_count)
        batch_weight = np.matmul(adj_matrix, norm_count)
        batch_weight = batch_weight / np.sum(batch_weight)

        return 1/(batch_weight+self.variance_control)

