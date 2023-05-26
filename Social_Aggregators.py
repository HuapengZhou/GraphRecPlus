import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
from Attention import Attention
import matplotlib.pyplot as plt

class Social_Aggregator(nn.Module):
    """
    Social Aggregator: for aggregating embeddings of social neighbors.
    """

    def __init__(self, features, u2e, embed_dim, cuda="cpu"):
        super(Social_Aggregator, self).__init__()

        self.features = features
        self.device = cuda
        self.u2e = u2e
        self.embed_dim = embed_dim
        self.att = Attention(self.embed_dim)

    def forward(self, nodes, to_neighs):
        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)
        att_weights = []  # 初始化一个列表来存储注意力权重
        for i in range(len(nodes)):
            tmp_adj = to_neighs[i]
            num_neighs = len(tmp_adj)
            # 
            e_u = self.u2e.weight[list(tmp_adj)] # fast: user embedding 
            #slow: item-space user latent factor (item aggregation)
            #feature_neigbhors = self.features(torch.LongTensor(list(tmp_adj)).to(self.device))
            #e_u = torch.t(feature_neigbhors)

            u_rep = self.u2e.weight[nodes[i]]

            att_w = self.att(e_u, u_rep, num_neighs)
            att_weights.append(att_w.detach().cpu().numpy())  # 将注意力权重添加到列表中
            att_history = torch.mm(e_u.t(), att_w).t()
            embed_matrix[i] = att_history
            # 绘制每个节点的注意力权重
            # plt.figure(figsize=(6, 6))
            #plt.imshow(att_w.detach().cpu().numpy(), cmap='hot', interpolation='nearest')  # 注意这里是绘制 att_w，而不是整个 att_weights 列表
            #plt.colorbar()
            #plt.title(f"Attention Weights for node {i}")
            #plt.show()
        to_feats = embed_matrix
        return to_feats
