import numpy as np
from .Abstract_Ranker import Abstract_Reweigher

r"""
Min-max-SGD
################################################


@InProceedings{pmlr-v162-abernethy22a,
  title = 	 {Active Sampling for Min-Max Fairness},
  author =       {Abernethy, Jacob D and Awasthi, Pranjal and Kleindessner, Matth{\"a}us and Morgenstern, Jamie and Russell, Chris and Zhang, Jie},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {53--65},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/abernethy22a/abernethy22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/abernethy22a.html},
  abstract = 	 {We propose simple active sampling and reweighting strategies for optimizing min-max fairness that can be applied to any classification or regression model learned via loss minimization. The key intuition behind our approach is to use at each timestep a datapoint from the group that is worst off under the current model for updating the model. The ease of implementation and the generality of our robust formulation make it an attractive option for improving model performance on disadvantaged groups. For convex learning problems, such as linear or logistic regression, we provide a fine-grained analysis, proving the rate of convergence to a min-max fair solution.}
}


"""

class Minmax_SGD(Abstract_Reweigher):
    def __init__(self, config, group_weight):
        super().__init__(config)
        self.group_weight = group_weight

    def reset_parameters(self):
        pass

    def reweight(self, input_dict):
        items = input_dict['items']
        losses = input_dict['loss'] #[B]

        adj_matrix = self.M[items]
        #loss_groups = np.mean(losses[:, np.newaxis] * adj_matrix, axis=0, keepdims=False)
        group_num = np.sum(adj_matrix, axis=0)
        loss_groups = np.sum(losses[:, np.newaxis] * adj_matrix, axis=0, keepdims=False) / (group_num + 1e-1)

        worst_groups = np.argmax(loss_groups)
        weights = np.zeros(self.config['group_num']) + self.config['p']
        weights[worst_groups] = 1.0 * (1-self.config['p'])
        batch_weight = np.matmul(adj_matrix, weights * self.group_weight)
        #batch_weight = batch_weight / np.sum(batch_weight)

        return batch_weight

