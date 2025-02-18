import numpy as np
from .Abstract_Ranker import Abstract_Regularizer
import torch.nn.functional as F
import torch
import torch.nn as nn

r"""
DPR
################################################
@inproceedings{10.1145/3397271.3401177,
author = {Zhu, Ziwei and Wang, Jianling and Caverlee, James},
title = {Measuring and Mitigating Item Under-Recommendation Bias in Personalized Ranking Systems},
year = {2020},
isbn = {9781450380164},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3397271.3401177},
doi = {10.1145/3397271.3401177},
booktitle = {Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {449–458},
numpages = {10},
keywords = {statistical parity, recommender systems, recommendation bias, equal opportunity},
location = {Virtual Event, China},
series = {SIGIR '20}
}

##Note that in this code, we extend the two-group cases into multi-group cases

"""

class DPR(Abstract_Regularizer):
    def __init__(self, config, group_weight):
        super().__init__(config)

        self.mlp = nn.Sequential(
            nn.Linear(self.config['embedding_size'], self.config['hidden_size']),  # 第一层
            nn.ReLU(),  # 第一层的激活函数
            nn.Linear(self.config['hidden_size'], self.config['group_num'])  # 第二层
        )

        self.group_loss = nn.CrossEntropyLoss()
        self.type = ["pair"]


    def fairness_loss(self, input_dict, Model):
        pos_items = input_dict['item_ids']
        neg_items = input_dict['neg_item_ids']

        pos_groups = input_dict['group_ids']
        neg_groups = input_dict['neg_group_ids']

        scores = input_dict['scores'] #[B]

        pos_item_embs = Model.item_embedding(pos_items)
        neg_item_embs = Model.item_embedding(neg_items)

        pos_group_predict = self.mlp(pos_item_embs)
        neg_group_predict = self.mlp(neg_item_embs)

        loss_pos_group = self.group_loss(pos_group_predict, pos_groups)
        loss_neg_group = self.group_loss(neg_group_predict, neg_groups)

        #group_scores =
        Q = torch.full_like(scores, 1.0 / self.config['item_num'])
        loss_uniform = scores * (torch.log(scores) - torch.log(Q))
        loss_uniform = torch.mean(loss_uniform)

        fairloss = -self.config['alpha'] * (loss_pos_group + loss_neg_group) + self.config['beta'] * loss_uniform

        return fairloss







