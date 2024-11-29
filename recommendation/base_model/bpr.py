import torch
import torch.nn as nn
import numpy as np
from .abstract_model import AbstractBaseModel


r"""
BPR
################################################
Reference:
    Steffen Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback." in UAI 2009.
"""

class BPR(AbstractBaseModel):
    def __init__(self, config):
        super().__init__(config)
        # self.user_embedding = nn.Embedding(num_users, embedding_dim)
        # self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.type = ["pair"]
        self.IR_type = ["retrieval", "ranking"]

        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)


    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        dot_product = (user_embeds * item_embeds).sum(1)
        return self.sigmoid(dot_product)

    def get_user_embedding(self, user_dict):
        user = user_dict['user_ids']
        return self.user_embedding(user)

    def compute_loss(self, interaction):
        #user = interaction['user_ids']
        pos_item = interaction['item_ids']
        neg_item = interaction['neg_item_ids']
        pos_score = self.forward(interaction, pos_item)
        neg_score = self.forward(interaction, neg_item)
        return self.bpr_loss(pos_score, neg_score)




