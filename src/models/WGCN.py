import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class WeightedGraphConv(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None, bias=True):
        super(WeightedGraphConv, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)
        self.activation = activation

    def forward(self, g, feat, eweight):
        with g.local_scope():
            # Store node features
            g.ndata['h'] = self.linear(feat)
            # Store edge weights
            g.edata['w'] = eweight

            # Message passing: weighted neighbor aggregation
            g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h'))

            h_new = g.ndata['h']
            if self.activation:
                h_new = self.activation(h_new)
            return h_new