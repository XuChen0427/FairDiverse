import torch
import torch.nn as nn
import numpy as np
from .abstract_model import AbstractBaseModel


r"""
BPR Sequential versions, since some dataset do not have the user id, so we utilize the mean history embedding to represent the user embeddings
################################################
"""

class BPR_Seq(AbstractBaseModel):
    def __init__(self, config):
        super().__init__(config)
        # self.user_embedding = nn.Embedding(num_users, embedding_dim)
        # self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.type = ["pair"]
        self.IR_type = ["retrieval", "ranking"]

        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)


    def forward(self, user_dict, item_ids):
        user_embeds = self.get_user_embedding(user_dict)
        item_embeds = self.item_embedding(item_ids)
        dot_product = (user_embeds * item_embeds).sum(1)
        return self.sigmoid(dot_product)

    def get_user_embedding(self, user_dict):
        #print(user_dict)
        user = user_dict['history_ids']
        user_embeds = self.item_embedding(user)  # [B, H, D]
        user_embeds = torch.mean(user_embeds, dim=1, keepdim=False)
        return user_embeds

    def compute_loss(self, interaction):
        #user = interaction['user_ids']
        pos_item = interaction['item_ids']
        neg_item = interaction['neg_item_ids']
        pos_score = self.forward(interaction, pos_item)
        neg_score = self.forward(interaction, neg_item)
        return self.bpr_loss(pos_score, neg_score)




