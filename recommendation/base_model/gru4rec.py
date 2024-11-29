import torch
import torch.nn as nn
import numpy as np
from .abstract_model import AbstractBaseModel


r"""
GRU4Rec
################################################

Reference:
    Yong Kiam Tan et al. "Improved Recurrent Neural Networks for Session-based Recommendations." in DLRS 2016.

"""

class GRU4Rec(AbstractBaseModel):
    def __init__(self, config):
        super().__init__(config)
        # self.user_embedding = nn.Embedding(num_users, embedding_dim)
        # self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.type = ["sequential"]
        self.IR_type = ["retrieval", "ranking"]

        self.emb_dropout = nn.Dropout(self.config['dropout_prob'])
        self.gru_layers = nn.GRU(
            input_size=self.config['embedding_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.config['hidden_size'], self.config['embedding_size'])

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.apply(self._init_weights)



    def compute_loss(self, interaction):

        #user = interaction['user_ids']
        #history_ids = interaction['history_ids']
        pos_items = interaction['item_ids']
        user_emb = self.get_user_embedding(interaction)
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(user_emb, test_item_emb.transpose(0, 1))
        loss = self.loss(logits, pos_items)
        return loss

    def get_user_embedding(self, user_dict):
        users = user_dict['history_ids']
        item_seq_emb = self.item_embedding(users)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        seq_output = gru_output[:,-1,:]
        #seq_output = self.gather_indexes(gru_output, self.config['history_length'] - 1)
        return seq_output



    def forward(self, user_dict, item_id):
        ###here user_id denotes the historical ids
        user_embeds = self.get_user_embedding(user_dict)
        item_embeds = self.item_embedding(item_id)
        dot_product = (user_embeds * item_embeds).sum(1)
        return self.sigmoid(dot_product)


