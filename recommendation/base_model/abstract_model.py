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


    def predict(self, user_id, item_id):
        ##used to re-rankiing tasks
        pass

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)