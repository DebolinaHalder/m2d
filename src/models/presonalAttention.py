import torch
import torch.nn as nn
import torch.nn.functional as F


class PersonaAttention(nn.Module):
    def __init__(self, num_pers: int, input_size: int):
        """
        Args:
            num_pers: number of personas (views)
            input_size: feature dimension
        """
        super().__init__()

        # Persona-specific feature scaling weights
        self.weight_tensor = nn.Parameter(
            torch.empty(num_pers, input_size)
        )
        nn.init.xavier_uniform_(self.weight_tensor)

        # Optional: used later for masking / thresholding
        self.markoff_value = 0.0

    def forward(self, context):
        """
        Args:
            context:
                - (N, F) or
                - (B, N, F)

        Returns:
            attention matrix:
                - (N, N) or
                - (B, N, N)
        """

        # Expand persona weights
        # (P, F) -> (P, 1, F)
        expand_weight = self.weight_tensor.unsqueeze(1)

        # If batched input, expand again
        if context.dim() == 3:
            # (P, 1, 1, F)
            expand_weight = expand_weight.unsqueeze(1)

        # Apply persona-specific scaling
        # 2D: (1, N, F) * (P, 1, F) -> (P, N, F)
        # 3D: (1, B, N, F) * (P, 1, 1, F) -> (P, B, N, F)
        context_fc = context.unsqueeze(0) * expand_weight

        # L2 normalize for cosine similarity
        context_norm = F.normalize(context_fc, p=2, dim=-1)

        # Persona-wise self-attention
        # (..., N, F) x (..., F, N) -> (..., N, N)
        attention = torch.matmul(
            context_norm,
            context_norm.transpose(-1, -2)
        )

        # Average over personas
        attention = attention.mean(dim=0)

        return torch.sigmoid(attention)