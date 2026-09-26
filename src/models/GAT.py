import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 feat_drop,
                 negative_slope):
        super().__init__()

        self.W = nn.Parameter(torch.empty(in_feats, out_feats))
        self.attn_l = nn.Parameter(torch.empty(out_feats, 1))
        self.attn_r = nn.Parameter(torch.empty(out_feats, 1))

        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)

        self.feat_drop = nn.Dropout(feat_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.alpha = None

    def forward(self, adj, x):

        N = adj.size(0)

        # self-loops
        A = adj + torch.eye(N, device=adj.device)

        x = self.feat_drop(x)

        h = x @ self.W

        el = h @ self.attn_l
        er = h @ self.attn_r

        e = self.leaky_relu(el + er.T)

        e = e.masked_fill(A <= 0, -1e15)
        e = e + torch.log(A + 1e-12)

        alpha = F.softmax(e, dim=1)
        self.alpha = alpha

        return alpha @ h


class GAT_body(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):

        super().__init__()

        self.layers = nn.ModuleList()

        # first layer
        self.layers.append(
            GraphAttentionLayer(
                in_dim,
                num_hidden,
                feat_drop,
                negative_slope,
            )
        )

        # hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(
                GraphAttentionLayer(
                    num_hidden,
                    num_hidden,
                    feat_drop,
                    negative_slope,
                )
            )

    def forward(self, adj, x):

        for i, layer in enumerate(self.layers):
            x = layer(adj, x)

            if i != len(self.layers) - 1:
                x = F.elu(x)

        return x

class GAT(nn.Module):
    def __init__(self,
                 nfeat,
                 nhid,
                 nclass,
                 dropout,
                 alpha=0.2):

        super().__init__()

        self.body = GAT_body(
            nfeat,
            nhid,
            dropout,
            alpha,
        )

        self.fc = nn.Linear(nhid, nclass)

    def forward(self, adj, x):

        h = self.body(adj, x)

        return self.fc(h)