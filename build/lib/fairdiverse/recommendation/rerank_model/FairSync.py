import os
import numpy as np
from .Abstract_Reranker import Abstract_Reranker
from tqdm import tqdm,trange
import torch

r"""
FairSync, a post-processing method to deal with minimum guarantee of different groups in retrieval process. 
Note that in this toolkit, in order to compare the performance fairly, we simplify it as the post-processing for ranking scores, 
which has the same effect of original paper.
 
The original FairSync is implemented in https://github.com/XuChen0427/FairSync/ using the faiss retrieval toolkit.

@inproceedings{Xu24FairSync,
author = {Xu, Chen and Xu, Jun and Ding, Yiming and Zhang, Xiao and Qi, Qi},
title = {FairSync: Ensuring Amortized Group Exposure in Distributed Recommendation Retrieval},
year = {2024},
isbn = {9798400701719},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3589334.3645413},
doi = {10.1145/3589334.3645413},
booktitle = {Proceedings of the ACM Web Conference 2024},
pages = {1092–1102},
numpages = {11},
keywords = {distributed retrieval, minimum exposures, recommender system},
location = {Singapore, Singapore},
series = {WWW '24}
}

"""

class FairSync(Abstract_Reranker):
    def __init__(self, config, weights = None):
        super().__init__(config, weights)


    def rerank(self, ranking_score, k):
        ## its parameters
        user_size = len(ranking_score)
        mu_t = torch.zeros(self.group_num, requires_grad=True)
        rerank_list = []
        minimum_exposure = self.config['minimum_exposure']
        optimizer = torch.optim.Adam([mu_t, ], lr=self.config['learning_rate'])

        for u in trange(user_size):
            loss = 0.0
            query = mu_t.cpu().detach().numpy()
            #print(query)
            rel = ranking_score[u,:] - np.matmul(self.M, query)
            result_item = np.argsort(rel)[::-1]
            result_item = result_item[:k]
            rerank_list.append(result_item)
            for i in result_item:
                pid = np.argmax(self.M[i])
                loss = loss + torch.tensor(ranking_score[u,i]) - mu_t[pid]
            #B_l = B_l + np.sum(self.M[result_item,:],axis=0,keepdims=False)
            loss += (torch.sum(minimum_exposure * mu_t) + torch.max(mu_t) * (user_size * k - minimum_exposure * self.config['group_num'])) / user_size
            optimizer.zero_grad()
            # loss =
            loss.backward()
            optimizer.step()

        return rerank_list
