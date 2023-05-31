import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import random
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, heads=1):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.to_keys = nn.Linear(embed_dim, embed_dim * heads, bias=False)
        self.to_queries = nn.Linear(embed_dim, embed_dim * heads, bias=False)
        self.to_values = nn.Linear(embed_dim, embed_dim * heads, bias=False)
        self.unifyheads = nn.Linear(heads * embed_dim, embed_dim)

    def forward(self, node1, u_rep, num_neighs):
        b, t, k = node1.size()
        u_rep = u_rep.repeat(b, 1).unsqueeze(1)

        keys = self.to_keys(node1).view(b, t, self.heads, k)
        queries = self.to_queries(u_rep).view(b, self.heads, k)
        values = self.to_values(node1).view(b, t, self.heads, k)

        keys = keys.transpose(1, 2).contiguous().view(b * self.heads, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b * self.heads, k)
        values = values.transpose(1, 2).contiguous().view(b * self.heads, t, k)

        queries = queries / (k ** (1 / 4))
        keys = keys / (k ** (1 / 4))

        att_w = torch.bmm(queries.unsqueeze(1), keys.transpose(1, 2))
        att_w = F.softmax(att_w, dim=2)
        out = torch.bmm(att_w, values).view(b, self.heads, t, k)
        out = out.transpose(1, 2).contiguous().view(b, t, self.heads * k)

        return self.unifyheads(out).squeeze(1)
