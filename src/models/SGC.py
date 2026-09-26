import torch
import torch.nn as nn



class SGC_Body(nn.Module):
    def __init__(self, nfeat, nhid, k=2):
        super(SGC_Body, self).__init__()

        self.k = k
        self.linear = nn.Linear(nfeat, nhid)

    def forward(self, adj, x):

        # Add self-loops
        I = torch.eye(
            adj.size(0),
            device=adj.device,
            dtype=adj.dtype
        )

        A = adj + I

        # Symmetric normalization
        deg = A.sum(dim=1)
        deg_inv_sqrt = (deg + 1e-12).pow(-0.5)

        A_hat = (
            deg_inv_sqrt.unsqueeze(1)
            * A
            * deg_inv_sqrt.unsqueeze(0)
        )

        # K-step propagation
        for _ in range(self.k):
            x = A_hat @ x

        # Learnable projection
        x = self.linear(x)

        return x


class SGC(nn.Module):
    def __init__(self, nfeat, nclass, k=2):
        super().__init__()

        self.body = SGC_Body(k=k)
        self.fc = nn.Linear(nfeat, nclass)

    def forward(self, adj, x):

        x = self.body(adj, x)

        return self.fc(x)