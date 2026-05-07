import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class SymmetricEdgePredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2*in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.LayerNorm(1)

        )

    def forward(self, H):
        n = H.size(0)
        h_i = H.unsqueeze(1).repeat(1, n, 1)   # [n, n, d]
        h_j = H.unsqueeze(0).repeat(n, 1, 1)   # [n, n, d]
        
        # Symmetric features
        diff = torch.abs(h_i - h_j)            # [n, n, d]
        prod = h_i + h_j                       # [n, n, d]
        z = torch.cat([diff, prod], dim=-1)    # [n, n, 2d]
        
        A_pred = (self.mlp(z).squeeze(-1))
        A_pred = torch.sigmoid(A_pred)       # [n, n]
        return A_pred