import os
import yaml
import json
import numpy as np
import torch
from ..utils import Build_Adjecent_Matrix



class Abstract_Reweigher(object):
    def __init__(self, config):
        self.config = config

        self.M, self.iid2pid = Build_Adjecent_Matrix(config)
        self.IR_type = ["ranking", "retrieval"]
        self.fair_type = "re-weight"
        self.type = ["point", "pair", "sequential"]
        #self.reset_parameters()

    # def reset_parameters(self):
    #     self.update_len = self.config['update_epoch'] * self.train_len
    #     self.C_t = self.config['topk'] * self.update_len * self.group_weight
    def reset_parameters(self, **kwargs):
        self.exposure_count = np.zeros(self.config['group_num'])


    def reweight(self, items):
        pass


class Abstract_Regularizer(object):
    def __init__(self, config):
        self.type = ["pair", "sequential"]
        self.IR_type = ["ranking", "retrieval"]
        self.fair_type = "regularizer"
        self.config = config
        self.M, self.iid2pid = Build_Adjecent_Matrix(config)

    def fairness_loss(self, **kwargs):
        pass

class Abstract_Sampler(object):
    def __init__(self, config, user2pos):
        self.user2pos = user2pos ###record the negative item corpus
        self.type = ["pair"]
        self.IR_type = ["ranking", "retrieval"]
        self.fair_type = "sample"
        self.config = config
        self.M, self.iid2pid = Build_Adjecent_Matrix(config)
        self.group_sizes = np.sum(self.M, axis=0)
        #print(self.group_sizes)
        #exit(0)

    def reset_parameters(self, **kwargs):
        self.sample_probality = np.ones(self.config['group_num'])
        self.sample_probality = self.sample_probality/np.sum(self.sample_probality)

    def sample(self, **kwargs):
        pass
