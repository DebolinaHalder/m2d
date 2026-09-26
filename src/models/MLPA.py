import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from torch.nn import Parameter
import torch

class MLPA(torch.nn.Module):

    def __init__(self, in_feats, dim_h, dim_z):
        super(MLPA, self).__init__()
        
        self.gcn_mean = torch.nn.Sequential(
                torch.nn.Linear(in_feats, dim_h),
                torch.nn.ReLU(),
                torch.nn.Linear(dim_h, dim_z)
                )
        

    def forward(self, hidden):
        # GCN encoder
        Z = F.normalize(self.gcn_mean(hidden))
        # inner product decoder
        adj_logits = Z @ Z.T
        return adj_logits