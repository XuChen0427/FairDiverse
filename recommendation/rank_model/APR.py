import numpy as np
from .Abstract_Ranker import Abstract_Reweigher

r"""
APR
################################################


@INPROCEEDINGS{10191757,
  author={Hu, Zhihao and Xu, Yiran and Tian, Xinmei},
  booktitle={2023 International Joint Conference on Neural Networks (IJCNN)}, 
  title={Adaptive Priority Reweighing for Generalizing Fairness Improvement}, 
  year={2023},
  volume={},
  number={},
  pages={01-08},
  keywords={Training;Adaptation models;Neural networks;Decision making;Machine learning;Predictive models;Benchmark testing},
  doi={10.1109/IJCNN54540.2023.10191757}
}



"""

class APR(Abstract_Reweigher):
    def __init__(self, config, group_weight):
        super().__init__(config)
        self.group_weight = group_weight

    def reset_parameters(self):
        self.sigma = np.zeros(self.config['group_num'])
        #self.subgroup_weights = np.ones(self.config['group_num']) * 0.1

    def reweight(self, input_dict):
        items = input_dict['items']
        losses = input_dict['loss'] #[B]

        adj_matrix = self.M[items]
        group_num = np.sum(adj_matrix, axis=0)
        loss_groups = np.sum(losses[:, np.newaxis] * adj_matrix, axis=0, keepdims=False)/(group_num+1e-1)
        #loss_groups = loss_groups / np.sum(loss_groups)

        #print(loss_groups)
        #exit(0)

        self.sigma = (1-self.config['beta']) + self.config['beta'] * (loss_groups-0.0)  #here since we utilize the loss, therefore d=0
        #print(self.sigma)
        B_t = np.sum(adj_matrix, axis=0, keepdims=False)
        #self.subgroup_weights = self.subgroup_weights * 1/((B_t+self.config['alpha'])/(np.sum(B_t)+self.config['alpha']))
        exp_sigma = 1/np.exp(-self.config['eta']*self.sigma)
        #weights = self.subgroup_weights * exp_sigma
        weights = exp_sigma
        batch_weight = np.matmul(adj_matrix, weights * self.group_weight)
        batch_weight = batch_weight / np.sum(batch_weight)
        #print(batch_weight)
        #exit(0)

        return batch_weight

