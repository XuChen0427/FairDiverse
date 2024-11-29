import numpy as np
from .Abstract_Ranker import Abstract_Sampler
import torch.nn.functional as F
import torch
import random

r"""
FairNeg
################################################
@inproceedings{10.1145/3543507.3583355,
author = {Chen, Xiao and Fan, Wenqi and Chen, Jingfan and Liu, Haochen and Liu, Zitao and Zhang, Zhaoxiang and Li, Qing},
title = {Fairly Adaptive Negative Sampling for Recommendations},
year = {2023},
isbn = {9781450394161},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3543507.3583355},
doi = {10.1145/3543507.3583355},
abstract = {Pairwise learning strategies are prevalent for optimizing recommendation models on implicit feedback data, which usually learns user preference by discriminating between positive (i.e., clicked by a user) and negative items (i.e., obtained by negative sampling). However, the size of different item groups (specified by item attribute) is usually unevenly distributed. We empirically find that the commonly used uniform negative sampling strategy for pairwise algorithms (e.g., BPR) can inherit such data bias and oversample the majority item group as negative instances, severely countering group fairness on the item side. In this paper, we propose a Fairly adaptive Negative sampling approach (FairNeg), which improves item group fairness via adaptively adjusting the group-level negative sampling distribution in the training process. In particular, it first perceives the model’s unfairness status at each step and then adjusts the group-wise sampling distribution with an adaptive momentum update strategy for better facilitating fairness optimization. Moreover, a negative sampling distribution Mixup mechanism is proposed, which gracefully incorporates existing importance-aware sampling techniques intended for mining informative negative samples, thus allowing for achieving multiple optimization purposes. Extensive experiments on four public datasets show our proposed method’s superiority in group fairness enhancement and fairness-utility tradeoff.},
booktitle = {Proceedings of the ACM Web Conference 2023},
pages = {3723–3733},
numpages = {11},
keywords = {BPR, Fairness, Negative Sampling., Recommender Systems},
location = {Austin, TX, USA},
series = {WWW '23}
}
"""


class FairNeg(Abstract_Sampler):
    def __init__(self, config, user2pos):
        super().__init__(config, user2pos)

        self.sample_probality = np.ones(self.config['group_num'])
        self.sample_probality = self.sample_probality / np.sum(self.sample_probality)

        self.history_gradient = np.zeros(self.config['group_num'])

        self.epoch_loss = np.zeros(self.config['group_num'])

    def reset_parameters(self):

        gradient = self.epoch_loss -  np.mean(self.epoch_loss)
        if np.sum(gradient) == 0:
            pass
        else:
            gradient = gradient / np.sum(gradient)
            v = self.config['gamma'] * self.history_gradient + self.config['alpha'] * gradient
            v = v/ np.sum(v)
            self.history_gradient = v
            self.sample_probality = self.sample_probality - v
            self.sample_probality = self.sample_probality / (np.sum(self.sample_probality)+1e-5)
            self.epoch_loss = np.zeros(self.config['group_num'])


    def accumulate_epoch_loss(self, delta_epoch_loss):
        self.epoch_loss += delta_epoch_loss


    def sample(self, interaction, Model):
        user_ids = interaction['user_ids'].cpu().numpy().tolist()
        item_ids = interaction['user_ids'].cpu().numpy()
        neg_items = []

        pos_items = []
        for u in user_ids:
            pos = self.user2pos[u]
            pos_items.extend(pos)
        pos_items = set(pos_items)
        neg_item_corpus = set(range(self.config['item_num'])) - pos_items
        neg_item_corpus = random.sample(list(neg_item_corpus), self.config['neg_sample_num'])


        group_fair = self.sample_probality / (self.group_sizes + 1)
        prob_fair = np.matmul(self.M[neg_item_corpus], group_fair)

        scores = Model.full_scores(interaction, torch.tensor(neg_item_corpus,dtype=torch.long).to(interaction['user_ids'].device))
        scores = torch.softmax(scores/self.config['temp'], dim=-1)
        scores = scores.detach().cpu().numpy()
        for i, user in enumerate(user_ids):

            prob = self.config['beta'] * prob_fair + (1-self.config['beta']) * scores[i]
            prob = prob/np.sum(prob+1e-5)
            #print(prob)
            neg_item = random.choices(neg_item_corpus, weights=prob, k=1)[0]
            neg_items.append(neg_item)

        return torch.tensor(neg_items, dtype=torch.long).to(interaction['user_ids'].device), self.M[item_ids]








