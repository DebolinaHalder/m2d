import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from torch.nn import Parameter
import torch
from dgl.nn import EdgeWeightNorm


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out




class GCNLdam(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCNLdam, self).__init__()
        self.body = GCN_Body(nfeat,nhid,dropout)
        self.fc = NormedLinear(nhid,nclass)

    def forward(self, g, x):
        x = self.body(g,x)
        x = self.fc(x)
        return x

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.body = GCN_Body(nfeat,nhid,dropout)
        self.fc = nn.Linear(nhid,nclass)

    def forward(self, g, x,edge_weights = None):
        x = self.body(g,x,edge_weights)
        x = self.fc(x)
        return x

# def GCN(nn.Module):
class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_Body, self).__init__()

        self.gc1 = GraphConv(nfeat, nhid,allow_zero_in_degree=True,norm = 'both')
        self.gc2 = GraphConv(nhid, nhid,allow_zero_in_degree=True,norm = 'both')
        #self.gc3 = GraphConv(nhid, nhid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x, edge_weights = None):
        #norm = EdgeWeightNorm(norm='none')
        #if edge_weights is not None:
        #    edge_weights = norm(g, edge_weights)
        x = F.relu(self.gc1(g, x, edge_weight=edge_weights))
        x = self.dropout(x)
        z = self.gc2(g, x, edge_weight=edge_weights)
        # x = self.dropout(x)
        return z    




