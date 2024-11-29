import numpy as np
from .Abstract_Ranker import Abstract_Reweigher

r"""
S-DRO
################################################

@inproceedings{10.1145/3485447.3512255,
author = {Wen, Hongyi and Yi, Xinyang and Yao, Tiansheng and Tang, Jiaxi and Hong, Lichan and Chi, Ed H.},
title = {Distributionally-robust Recommendations for Improving Worst-case User Experience},
year = {2022},
isbn = {9781450390965},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3485447.3512255},
doi = {10.1145/3485447.3512255},
abstract = {Modern recommender systems have evolved rapidly along with deep learning models that are well-optimized for overall performance, especially those trained under Empirical Risk Minimization (ERM). However, a recommendation algorithm that focuses solely on the average performance may reinforce the exposure bias and exacerbate the “rich-get-richer” effect, leading to unfair user experience. In a simulation study, we demonstrate that such performance gap among various user groups is enlarged by an ERM-trained recommender in the long-term. To mitigate such amplification effects, we propose to optimize for the worst-case performance under the Distributionally Robust Optimization (DRO) framework, with the goal of improving long-term fairness for disadvantaged subgroups. In addition, we propose a simple-yet-effective streaming optimization improvement called Streaming-DRO (S-DRO), which effectively reduces loss variances for recommendation problems with sparse and long-tailed data distributions. Our results on two large-scale datasets suggest that (1) DRO is a flexible and effective technique for improving worst-case performance, and (2) Streaming-DRO outperforms vanilla DRO and other strong baselines by improving the worst-case and overall performance at the same time.},
booktitle = {Proceedings of the ACM Web Conference 2022},
pages = {3606–3610},
numpages = {5},
keywords = {Distributional robustness, Recommendation., Robust learning},
location = {Virtual Event, Lyon, France},
series = {WWW '22}
}


"""

class SDRO(Abstract_Reweigher):
    def __init__(self, config, group_weight):
        super().__init__(config)
        self.group_weight = group_weight

    def reset_parameters(self):
        self.mu = np.ones(self.config['group_num'])

    def reweight(self, input_dict):
        items = input_dict['items']
        losses = input_dict['loss'] #[B]

        adj_matrix = self.M[items]

        #C_t = np.mean(self.M[items], axis=1, keepdims=False) #[B,G]
        group_num = np.sum(adj_matrix, axis=0)
        gradient = np.sum(losses[:, np.newaxis] * adj_matrix, axis=0, keepdims=False) / (group_num + 1e-1)
        ###since the loss is the negative of the scores
        self.mu = self.mu + self.config['eta'] * gradient
        #self.mu = np.clip(self.mu, 1e-3, 1)
        self.mu = np.exp(self.mu)
        self.mu = self.mu / np.sum(self.mu)

        batch_weight = np.matmul(adj_matrix, self.mu * self.group_weight)
        batch_weight = batch_weight / np.sum(batch_weight)


        return batch_weight

