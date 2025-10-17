import torch
import torch.nn as nn
import torch.nn.functional as nn_functional
import argparse


class DynamicBranchAttention(nn.Module):
    def __init__(self, N, T, hidden_dim=16):
        super().__init__()
        """
        Args:
            N: Number of features (e.g., 2)
            T: Number of timesteps (e.g., 5)
            hidden_dim: Hidden dimension size
        """
        self.N = N
        self.T = T

        # Map each branch's global features (N*T) to a scalar score
        self.score_net = nn.Sequential(
            nn.Linear(N * T, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output single score
        )

    def forward(self, out1, out2, out3, out4, out5):
        """
        Args:
            out1~out5: shape (B, H, W, N, T)
        Returns:
            attn: shape (B, 5) - Weights for each branch
        """
        B = out1.shape[0]
        outs = [out1, out2, out3, out4, out5]
        scores = []  # Collect scores for each branch (B, 1)

        for out in outs:
            # Global average pooling: compress spatial dimensions H, W
            z = out.mean(dim=(1, 2))  # (B, N, T)
            z = z.view(B, -1)  # (B, N*T)
            s = self.score_net(z)  # (B, 1)
            scores.append(s)  # Each is (B, 1)

        # Concatenate into (B, 5)
        scores = torch.cat(scores, dim=1)  # (B, 5)

        # Softmax normalization to get attention weights
        attn = nn_functional.softmax(scores, dim=1)  # (B, 5)

        return attn  # ✅ Output is 5 weights!