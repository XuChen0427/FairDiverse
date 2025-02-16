import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import normal_

class AbstractBaseModel(nn.Module):
    def __init__(self, config):
        super(AbstractBaseModel, self).__init__()

        self.config = config

        self.user_embedding = nn.Embedding(config['user_num'], config['embedding_size'])
        self.item_embedding = nn.Embedding(config['item_num'], config['embedding_size'])
        self.sigmoid = nn.Sigmoid()

    #def interaction2user(self,interaction):

    def forward(self, **kwargs):
        pass
        # user_embeds = self.user_embedding(user_ids)
        # item_embeds = self.item_embedding(item_ids)
        # dot_product = (user_embeds * item_embeds).sum(1)
        # return dot_product

    def compute_loss(self, **kwargs):
        pass

    def get_user_embedding(self, user):
        return self.user_embedding(user)

    def full_predict(self, user_dict, items):
        # here we assume only one user arrives

        #user = torch.unsqueeze(user)
        #items = torch.unsqueeze(items, 0)

        user_embeds = self.get_user_embedding(user_dict)
        item_embeds = self.item_embedding(items)
        scores = (user_embeds * item_embeds).sum(-1)
        return self.sigmoid(scores)

    def full_ranking(self, user_dict, items, k):
        user_embeds = self.get_user_embedding(user_dict) #[B, D]
        item_embeds = self.item_embedding(items)  # [H, D]
        scores = torch.matmul(user_embeds, item_embeds.t()) #[B,H]
        scores, indices = torch.topk(scores, k=k, dim=-1)
        return scores, indices

    def full_scores(self, user_dict, items):
        user_embeds = self.get_user_embedding(user_dict)  # [B, D]
        item_embeds = self.item_embedding(items)  # [H, D]
        scores = torch.matmul(user_embeds, item_embeds.t())  # [B,H]
        return scores

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def bpr_loss(self, pos_score, neg_score):
        return -torch.log(1e-10 + torch.sigmoid(pos_score - neg_score))