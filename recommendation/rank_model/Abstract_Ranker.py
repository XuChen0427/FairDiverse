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
        self.reset_parameters()

    # def reset_parameters(self):
    #     self.update_len = self.config['update_epoch'] * self.train_len
    #     self.C_t = self.config['topk'] * self.update_len * self.group_weight
    def reset_parameters(self):
        self.exposure_count = np.zeros(self.config['group_num'])


    def reweight(self, items):
        pass