import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from torch.nn import Parameter
import torch
from dgl.nn import EdgeWeightNorm


import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, adj, x):
        """
        adj: (N, N) dense adjacency matrix
        x:   (N, F)
        """

        # Add self-loops
        A = adj + torch.eye(adj.size(0), device=adj.device)

        # Symmetric normalization
        deg = A.sum(dim=1)
        deg_inv_sqrt = (deg + 1e-12).pow(-0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)

        A_hat = D_inv_sqrt @ A @ D_inv_sqrt

        return A_hat @ x @ self.weight

class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super().__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, adj, x):
        x = F.relu(self.gc1(adj, x))
        x = self.dropout(x)
        x = self.gc2(adj, x)
        return x

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()

        self.body = GCN_Body(nfeat, nhid, dropout)
        self.fc = nn.Linear(nhid, nclass)

    def forward(self, adj, x):
        x = self.body(adj, x)
        return self.fc(x) 




