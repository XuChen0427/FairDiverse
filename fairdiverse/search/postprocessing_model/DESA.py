from collections import *
import torch
import torch.nn.init as init
from torch import nn

from .base import BasePostProcessModel

MAXDOC = 50

'''
@inproceedings{qin2020diversifying,
  title={Diversifying search results using self-attention network},
  author={Qin, Xubo and Dou, Zhicheng and Wen, Ji-Rong},
  booktitle={Proceedings of the 29th ACM International Conference on Information \& Knowledge Management},
  pages={1265--1274},
  year={2020}
}
'''


class SelfAttnEnc(nn.Module):
    def __init__(self, d_model, nhead, nlayers, dropout):
        """
        Initialization
        
        :param d_model: The dimension of the model.
        :param nhead: The number of attention heads.
        :param nlayers: The number of transformer encoder layers.
        :param dropout: Dropout probability.
        """
        super(SelfAttnEnc, self).__init__()
        self.enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=400, dropout=dropout,
                                                    batch_first=True)
        self.enc = nn.TransformerEncoder(self.enc_layer, num_layers=nlayers)

    def forward(self, input, mask):
        """
        Forward pass of the Self-Attention Encoder.

        :param input: Input tensor of shape (batch_size, sequence_length, d_model).
        :param mask: Mask tensor indicating padded positions.
        :return: Encoded output tensor of shape (batch_size, sequence_length, d_model).
        """
        enc_out = self.enc(input, src_key_padding_mask=mask) # input [bs , seq_len , d_model]
        return enc_out # enc_out [bs , seq_len , d_model]


class DESA(BasePostProcessModel):
    def __init__(self, doc_d_model, doc_nhead, doc_nlayers, sub_d_model, sub_nhead, sub_nlayers, nhead, dropout):
        """
        Initialization

        :param doc_d_model: Document embedding dimension.
        :param doc_nhead: Number of attention heads for document encoder.
        :param doc_nlayers: Number of transformer layers for document encoder.
        :param sub_d_model: Sub-document embedding dimension.
        :param sub_nhead: Number of attention heads for sub-document encoder.
        :param sub_nlayers: Number of transformer layers for sub-document encoder.
        :param nhead: Number of attention heads for multi-head attention.
        :param dropout: Dropout probability.
        """
        super().__init__(dropout)
        
        Linear_out = 128
        self.linear1 = nn.Linear(doc_d_model, Linear_out)
        self.linear2 = nn.Linear(sub_d_model, Linear_out)
        self.linear3 = nn.Linear(18, 1)
        self.linear4 = nn.Linear(18+Linear_out+Linear_out+10, 1)
        self.doc_attn = SelfAttnEnc(Linear_out, doc_nhead, doc_nlayers, dropout) # [bs , seq_len , d_model]
        self.sub_attn = SelfAttnEnc(Linear_out, sub_nhead, sub_nlayers, dropout) # [bs , seq_len , d_model]
        self.dec_attn = nn.MultiheadAttention(Linear_out, nhead, dropout=dropout, batch_first=True)

    def fit(self, doc_emb, sub_emb, doc_mask, sub_mask, pos_qrel_feat, pos_subrel_feat, index_i=None, index_j=None,
                neg_qrel_feat=None, neg_subrel_feat=None, subrel_mask=None, mode='Train'):
        """
        Model training.

        :param doc_emb: Document embeddings of shape (batch_size, sequence_length, embedding_dim).
        :param sub_emb: Sub-document embeddings of shape (batch_size, sequence_length, embedding_dim).
        :param doc_mask: Mask tensor for document sequences.
        :param sub_mask: Mask tensor for sub-document sequences.
        :param pos_qrel_feat: Positive query relevance features.
        :param pos_subrel_feat: Positive sub-document relevance features.
        :param index_i: Index tensor for selecting positive samples.
        :param index_j: Index tensor for selecting negative samples.
        :param neg_qrel_feat: Negative query relevance features (optional).
        :param neg_subrel_feat: Negative sub-document relevance features (optional).
        :param subrel_mask: Mask tensor for sub-document relevance (optional).
        :param mode: Mode of operation ('Train' or 'Eval').
        :return: Positive and negative ranking scores in training mode, or final scores in evaluation mode.
        """
        doc_mask, sub_mask = doc_mask.bool(), sub_mask.bool()
        doc_rep = self.doc_attn(self.linear1(doc_emb), doc_mask)  # [bs, sq(50), d_model]
        sub_rep = self.sub_attn(self.linear2(sub_emb), sub_mask)  # [bs, sq(10), d_model]
        doc_dec, _ = self.dec_attn(doc_rep, sub_rep, sub_rep) # [bs, sq(50), d_model]
        device = doc_rep.device
        if mode == 'Train':
            pos_index_select1 = torch.index_select(doc_rep.reshape((-1, doc_rep.shape[2])), 0,
                                                   (index_i.to(device) + torch.linspace(0, doc_rep.shape[0] - 1,
                                                doc_rep.shape[0]).to(device) * torch.tensor(
                                                       doc_rep.shape[0] + 1).to(device)).long())
            pos_index_select2 = torch.index_select(doc_dec.reshape((-1, doc_dec.shape[2])), 0,
                                                   (index_i.to(device) + torch.linspace(0, doc_dec.shape[0] - 1,
                                                   doc_dec.shape[0]).to(device) * torch.tensor(
                                                       doc_dec.shape[0] + 1).to(device)).long())
            pos_concat = torch.cat([pos_qrel_feat, pos_index_select1, pos_index_select2,
                                    self.linear3(pos_subrel_feat).squeeze(2)], dim=1)  # pos_subrel[bs, sq(10), 18]
            pos_out = self.linear4(pos_concat)
            neg_index_select1 = torch.index_select(doc_rep.reshape((-1, doc_rep.shape[2])), 0,
                    (index_j.to(device) + torch.linspace(0, doc_rep.shape[0] - 1, doc_rep.shape[0]).to(device) *
                     torch.tensor(doc_rep.shape[0] + 1).to(device)).long())
            neg_index_select2 = torch.index_select(doc_dec.reshape((-1, doc_dec.shape[2])), 0,
                    (index_j.to(device) + torch.linspace(0, doc_dec.shape[0] - 1, doc_dec.shape[0]).to(device) * torch.tensor(
                                                                doc_dec.shape[0] + 1).to(device)).long())

            neg_concat = torch.cat([neg_qrel_feat, neg_index_select1,
                                    neg_index_select2, self.linear3(neg_subrel_feat).squeeze(2)], dim=1)
            neg_out = self.linear4(neg_concat)
            return pos_out, neg_out
        else:
            pos_concat = torch.cat([pos_qrel_feat, doc_rep, doc_dec,
                                    self.linear3(pos_subrel_feat).squeeze(3)], dim=2)  # pos_subrel[bs, sq(10), 18]
            pos_out = self.linear4(pos_concat)
            return pos_out.squeeze(2).squeeze(0) # [50]


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP).
    """
    def __init__(self, input_size, hid_size, output_size):
        """
        Initialization

        :param input_size: The size of the input feature vector.
        :param hid_size: The number of hidden units.
        :param output_size: The number of output units.
        """
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, output_size)
        )

    def forward(self, input):
        """
        Forward of the MLP.

        :param input: Input feature vector.
        :return: Output tensor.
        """
        output = self.mlp(input)
        return output
