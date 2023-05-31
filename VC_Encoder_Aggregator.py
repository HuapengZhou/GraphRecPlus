import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from Attention import Attention

class VC_Aggregator(nn.Module):
    """
    item and category aggregator: for aggregating embeddings of neighbors (item/category aggregator).
    """

    def __init__(self, c2e, r2e, v2e, embed_dim, cuda="cpu", vc=True):
        super(VC_Aggregator, self).__init__()
        self.vc = vc
        self.c2e = c2e
        self.r2e = r2e
        self.v2e = v2e
        self.device = cuda
        self.embed_dim = embed_dim
        self.w_r1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att = Attention(self.embed_dim)

    def forward(self, nodes, history_vc, history_r):

        embed_matrix = torch.empty(len(history_vc), self.embed_dim, dtype=torch.float).to(self.device)

        for i in range(len(history_vc)):
            history = history_vc[i]
            num_histroy_item = len(history)
            tmp_label = history_r[i]

            history = torch.LongTensor(history).to(self.device)

            self.c2e = self.c2e.to(self.device)

            if self.vc == True:
                # item component
                e_vc = self.c2e.weight[history]
                vc_rep = self.v2e.weight[nodes[i]]
            else:
                # category component
                e_vc = self.v2e.weight[history]
                vc_rep = self.c2e.weight[nodes[i]]

            e_r = self.r2e.weight[tmp_label]
            x = torch.cat((e_vc, e_r), 1)
            x = F.relu(self.w_r1(x))
            o_history = F.relu(self.w_r2(x))

            att_w = self.att(o_history, vc_rep, num_histroy_item)
            att_history = torch.mm(o_history.t(), att_w)
            att_history = att_history.t()

            embed_matrix[i] = att_history
        to_feats = embed_matrix
        return to_feats

class VC_Encoder(nn.Module):

    def __init__(self, features, embed_dim, history_vc_lists, history_r_lists, aggregator, cuda="cpu", vc=True):
        super(VC_Encoder, self).__init__()

        self.features = features
        self.vc = vc
        self.history_vc_lists = history_vc_lists
        self.history_r_lists = history_r_lists
        self.aggregator = aggregator
        self.embed_dim = embed_dim
        self.device = cuda
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)

    def forward(self, nodes):
        tmp_history_vc = []
        tmp_history_r = []
        for node in nodes:
            tmp_history_vc.append(self.history_vc_lists[int(node)])
            tmp_history_r.append(self.history_r_lists[int(node)])

        neigh_feats = self.aggregator.forward(nodes, tmp_history_vc, tmp_history_r)  # item-category network

        self_feats = self.features.weight[nodes]
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))

        return combined
