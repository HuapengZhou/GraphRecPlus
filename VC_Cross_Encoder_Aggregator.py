import torch
import torch.nn as nn
from torch.nn import init
from CrossAttention import CrossAttention
import numpy as np
import random
import torch.nn.functional as F

class VC_Aggregator(nn.Module):
    """
    item and category aggregator: for aggregating embeddings of neighbors (item/category aggregator).
    """

    def __init__(self, c2e, r2e, v2e, embed_dim, heads=1, cuda="cpu", vc=True):
        super(VC_Aggregator, self).__init__()
        self.vc = vc
        self.c2e = c2e
        self.r2e = r2e
        self.v2e = v2e
        self.device = cuda
        self.embed_dim = embed_dim
        self.heads = heads
        self.att = CrossAttention(embed_dim, heads)

    def forward(self, nodes, history_vc, history_r):
        nodes = torch.LongTensor(nodes).to(self.device)
        embed_matrix = torch.empty(len(history_vc), self.embed_dim).to(self.device)

        for i in range(len(history_vc)):
            history = history_vc[i]
            num_histroy_item = len(history)
            tmp_label = history_r[i]
            history = torch.LongTensor(history).to(self.device)

            if self.vc == True:
                e_vc = self.c2e.weight[history]
                vc_rep = self.v2e.weight[nodes[i]]
            else:
                e_vc = self.v2e.weight[history]
                vc_rep = self.c2e.weight[nodes[i]]

            e_r = self.r2e.weight[tmp_label]
            x = torch.cat((e_vc, e_r), 1)
            x = x.view(num_histroy_item, 1, self.embed_dim * 2)

            att_history = self.att(x, vc_rep.unsqueeze(0), num_histroy_item)

            embed_matrix[i] = att_history

        return embed_matrix


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

        neigh_feats = self.aggregator.forward(nodes, tmp_history_vc, tmp_history_r)

        self_feats = self.features.weight[nodes]
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))

        return combined
