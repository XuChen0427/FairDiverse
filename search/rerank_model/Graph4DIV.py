from collections import *
import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
from modules import GraphConvolutionNetwork
MAXDOC = 50


class Graph4Div(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, activation = "relu", nodenum = 50, batch_norm = True, normalization = False, dropout = 0.1):
        super(Graph4Div, self).__init__()
        self.gcn_layers_dim = [node_feature_dim] + hidden_dim
        self.gcn_layers_num = len(self.gcn_layers_dim)
        print('dim = ', self.gcn_layers_dim)
        print('num = ', self.gcn_layers_num)
        self.input_dim = node_feature_dim
        self.output_dim = output_dim
        self.batch_norm = batch_norm
        self.node_num = nodenum
        self.normalization = normalization
        print('activation = {}, node_num = {}'.format(activation, self.node_num))
        print('Graph4DIV normalization\t', normalization)
        self.gnn = GraphConvolutionNetwork(node_feature_dim, hidden_dim, output_dim, activation, batch_norm, normalization)
        self.fc1 = nn.Linear(3 * self.output_dim + 1, 50)
        self.fc2 = nn.Linear(50, 1)
        self.nfc1 = nn.Linear(18, 18)
        self.nfc2 = nn.Linear(18, 8)
        self.nfc3 = nn.Linear(8, 1)
        init.xavier_normal_(self.nfc1.weight)
        init.xavier_normal_(self.nfc2.weight)
        init.xavier_normal_(self.nfc3.weight)
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        self.dropout = nn.Dropout(dropout)
        return
    
    def forward(self, A, x, rel_feat, degree_feat, pos_mask = None, neg_mask = None, mask_flag = False):
        h, x = self.gnn(A, x)
        h = h.reshape(h.shape[0], 1, h.shape[1])
        h = h.repeat(1, x.shape[1] - 1, 1)
        context_tensor = x[:, 0, :].unsqueeze(dim = 1).repeat(1, x.shape[1] - 1, 1)
        x = x[:, 1:, :]
        x = torch.cat([ x, h , context_tensor, degree_feat], dim = 2)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(x.shape[0] * self.node_num, 1)

        rel_feat = rel_feat.reshape(rel_feat.shape[0] * self.node_num, 18)
        sr = F.relu(self.nfc1(rel_feat))
        sr = F.relu(self.nfc2(sr))
        sr = self.nfc3(sr)

        x = x.view(x.shape[0] * x.shape[1])
        sr = sr.view(sr.shape[0] * sr.shape[1])
        x = 0.5*sr+0.5*x
        if mask_flag:
            pos_mask = pos_mask.view(pos_mask.shape[0] * pos_mask.shape[1])
            score_1 = torch.masked_select(x, pos_mask)
            neg_mask = neg_mask.view(neg_mask.shape[0] * neg_mask.shape[1])
            score_2 = torch.masked_select(x, neg_mask)
            return score_1, score_2
        else:
            return x