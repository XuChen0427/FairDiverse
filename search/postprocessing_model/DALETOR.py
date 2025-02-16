from collections import *
import torch
import math
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F
from torch.nn import LayerNorm
import numpy as np

from .base import BasePostProcessModel

MAXDOC=50

'''
@inproceedings{yan2021diversification,
  title={Diversification-aware learning to rank using distributed representation},
  author={Yan, Le and Qin, Zhen and Pasumarthi, Rama Kumar and Wang, Xuanhui and Bendersky, Michael},
  booktitle={Proceedings of the Web Conference 2021},
  pages={127--136},
  year={2021}
}
'''


class MHSA(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) module.

    :param H: The number of attention heads. Default is 2.
    :param input_dim: The input feature dimension. Default is 100.
    :param output_dim: The output feature dimension. Default is 256.
    """
    def __init__(self, H = 2, input_dim = 100, output_dim = 256):
        super(MHSA, self).__init__()
        self.head_num = H
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = LayerNorm(self.output_dim, eps=1e-5)
        self.d_k = math.sqrt(output_dim)
        self.W_Q = nn.ModuleList()
        self.W_K = nn.ModuleList()
        self.W_V = nn.ModuleList()
        self.W_out = nn.Linear(self.output_dim*self.head_num, self.output_dim)
        for i in range(self.head_num):
            self.W_Q.append(nn.Linear(self.input_dim, self.output_dim, False))
            self.W_K.append(nn.Linear(self.input_dim, self.output_dim, False))
            self.W_V.append(nn.Linear(self.input_dim, self.output_dim, False))
        for hid in range(self.head_num):
            init.xavier_normal_(self.W_Q[hid].weight)
            init.xavier_normal_(self.W_K[hid].weight)
            init.xavier_normal_(self.W_V[hid].weight)
        init.xavier_normal_(self.W_out.weight)
    
    def attention(self, Q, K, V):
        """
        Computes scaled dot-product attention.

        :param Q: Query matrix.
        :param K: Key matrix.
        :param V: Value matrix.
        :return: Attention-weighted output.
        """
        scores = torch.bmm(Q, K.transpose(-2,-1)) / self.d_k
        attn = F.softmax(scores, dim = -1)
        return torch.bmm(attn, V)
    
    def forward(self, D):
        """
        Forward pass for the MHSA module.

        :param D: Input tensor of shape (batch_size, sequence_length, input_dim).
        :return: Output tensor after multi-head self-attention and normalization.
        """
        X_list=[]
        for hid in range(self.head_num):
            Q = self.W_Q[hid](D)
            K = self.W_K[hid](D)
            V = self.W_V[hid](D)
            X = self.attention(Q, K, V)
            X_list.append(X)
        res = torch.concat(X_list, dim=-1).float()
        res = self.W_out(res)
        if self.input_dim == self.output_dim:
            res = self.norm(D + res)
        else:
            res = self.norm(res)
        return res


class DALETOR(BasePostProcessModel):
    """
    DALETOR model implementation

    :param dropout: Dropout rate for regularization. Default is 0.1.
    """
    def __init__(self, dropout = 0.1):
        super().__init__(dropout)

        self.activation = "relu"
        self.batch_norm = True
        self.normalization = False
        self.layer_num = 2
        self.feat_dims = [100, 256]
        
        self.DIN1 = MHSA(2, 100, 256)
        self.DIN2 = MHSA(2, 256, 256)
        
        self.fc1 = nn.Linear(556, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.batch_norm1 = nn.BatchNorm1d(50)
        self.batch_norm2 = nn.BatchNorm1d(50)
        self.batch_norm3 = nn.BatchNorm1d(50)


        self.nfc1 = nn.Linear(18,18)
        self.nfc2 = nn.Linear(18,8)
        self.nfc3 = nn.Linear(8,1)

        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fc3.weight)
        init.xavier_normal_(self.fc4.weight)

        init.xavier_normal_(self.nfc1.weight)
        init.xavier_normal_(self.nfc2.weight)
        init.xavier_normal_(self.nfc3.weight)


    def fit(self, x, rel_feat, train_flag = False):
        """
        Training stage

        :param x: Input tensor of shape (batch_size, sequence_length, feature_dim).
        :param rel_feat: Relevance feature tensor of shape (batch_size, num_docs, 18).
        :param train_flag: Boolean flag indicating whether the model is in training mode.
        :return: Ranking scores for documents.
        """
        bs = x.shape[0]
        seq_len = x.shape[1]
        df = x.shape[2]
        query_x = x[:,0,:].unsqueeze(1)
        doc_x = x[:,1:,:]
        C = query_x * doc_x
        a = self.DIN1(C)
        a = self.DIN2(a)

        query_x = query_x.repeat(1,50,1)
        feat = torch.cat([query_x, doc_x, C, a], dim=2).float()

        s = F.relu(self.batch_norm1(self.fc1(feat)))
        s = F.relu(self.batch_norm2(self.fc2(s)))
        s = F.relu(self.batch_norm3(self.fc3(s)))
        s = self.fc4(s)
        s = s.squeeze()

        rel_feat = rel_feat.reshape(rel_feat.shape[0] * rel_feat.shape[1], 18)
        sr = F.relu(self.nfc1(rel_feat))
        sr = F.relu(self.nfc2(sr))
        sr = self.nfc3(sr)
        sr = sr.squeeze()
        sr = sr.reshape(bs, MAXDOC)

        score = s + sr
        
        if train_flag:
            return score
        else:
            return score
