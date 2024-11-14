import torch
import torch.nn as nn
import numpy as np
from .abstract_model import AbstractBaseModel


r"""
MF
################################################
Reference:
    Matrix Factorization Models for learned embeddings of users and items
"""

class MF(AbstractBaseModel):
    def __init__(self, config):
        super().__init__(config)
        # self.user_embedding = nn.Embedding(num_users, embedding_dim)
        # self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.type = ["point"]
        self.IR_type = ["retrieval", "ranking"]

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss(reduction='none')
        self.apply(self._init_weights)


    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        dot_product = (user_embeds * item_embeds).sum(1)
        return self.sigmoid(dot_product)


    def compute_loss(self, interaction):
        user = interaction['user_ids']
        item = interaction['item_ids']
        label = interaction['label']
        output = self.forward(user, item)
        return self.loss(output, label)

    def full_predict(self, user, items):
        user_embeds = self.user_embedding(user) #[B, D]
        user_embeds = torch.unsqueeze(user_embeds, 1)
        item_embeds = self.item_embedding(items) #[B, H, D]
        scores = (user_embeds * item_embeds).sum(-1)
        return self.sigmoid(scores)



