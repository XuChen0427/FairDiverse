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

    def reset_parameters(self, **kwargs):
        self.exposure_count = np.zeros(self.config['group_num'])


    def reweight(self, **kwargs):
        """
            Recalculates the batch weights based on the loss and the group adjacency matrix.

            This function computes new weights for each group in the batch based on the loss associated with each item and
            the group adjacency matrix. The batch weights are recalculated by considering the loss for each group and adjusting
            them based on predefined parameters in the configuration (e.g., `beta`, `alpha`, `eta`).

            :param Any:
            :return: A normalized vector of batch weights for each group in the batch.
        """
        pass


class Abstract_Regularizer(object):
    def __init__(self, config):
        self.type = ["pair", "sequential"]
        self.IR_type = ["ranking", "retrieval"]
        self.fair_type = "regularizer"
        self.config = config
        self.M, self.iid2pid = Build_Adjecent_Matrix(config)

    def fairness_loss(self, **kwargs):
        """
           Computes the fairness loss as the variance of the input scores.

           This function calculates the fairness loss by computing the variance of the provided scores. The loss aims to capture
           the variation in the scores, which can be used as a measure of fairness in the model's predictions.

           :param: any.
           :return: The fairness loss.
           """
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
        """
        Samples negative items for each user in the interaction, based on group fairness and model predictions.

        This function selects negative items for users in the interaction based on two factors:
        group fairness (weighted by `group_fair`) and the model's predicted scores (scaled by a temperature parameter).
        The negative item selection combines these factors and returns the selected items.

        :param interaction: A dictionary containing interaction data. It should include:
                            - 'user_ids': A tensor of user IDs.
                            - 'item_ids': A tensor of item IDs.
        :param Model: The model object that provides the method `full_scores` to get predicted scores for negative items.
        :return: A tensor of sampled negative item IDs for each user, and the positive item embeddings.
        """


        pass
