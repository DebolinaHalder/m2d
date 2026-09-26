import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from torch.nn import Parameter
import torch



class MLPX(torch.nn.Module):

    def __init__(self, in_feats, n_hidden, out_feats, dropout = 0.2):
        super(MLPX, self).__init__()

        #self.dropout = nn.Dropout(dropout)

        self.fc1 = torch.nn.Linear(in_feats, n_hidden)
        #self.norm1 = torch.nn.LayerNorm(n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, out_feats)
        self.norm2 = torch.nn.LayerNorm(out_feats)
    
    def projection(self, z):
        z = torch.relu((self.fc1(z)))
        #return (self.norm2(self.fc2(z)))
        return ((self.fc2(z)))

    def forward(self, h):
        #return (self.projection(h))
        return torch.sigmoid(self.projection(h))