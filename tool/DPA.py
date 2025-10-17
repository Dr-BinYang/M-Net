import torch
import torch.nn as nn
import torch.nn.functional as nn_functional
import argparse

class DPA(nn.Module):
    """
    Dual Natural Patch Attention (DPA) Module
    Suitable for 5-stage encoder outputs of MNet, performs Patch Channel and Patch Spatial attention processing
    Input: encoder_output1 ~ encoder_output5
    Output: D1 ~ D5, with shapes matching the input, for skip connections
    """

    def __init__(self, channels_list, sizes_list, min_P=256):
        """
        Args:
            channels_list: list of int, channel numbers for each stage, e.g., [64, 128, 256, 512, 1024]
            sizes_list: list of (H_i, W_i), spatial dimensions for each stage
            min_P: int, unified patch count (for attention computation), recommended 256 or 512
        """
        super().__init__()
        self.num_stages = len(channels_list)
        self.channels_list = channels_list
        self.sizes_list = sizes_list
        self.min_P = min_P  # Unified P dimension

        # 1. Tokenization: FFN for each stage
        self.ffn_tokens = nn.ModuleList([
            nn.Sequential(
                nn.Linear(H_i * W_i, min_P),
            ) for (H_i, W_i) in sizes_list
        ])

        # 2. PCA: Patch Channel Attention
        self.q_proj = nn.ModuleList([
            nn.Conv1d(C, C, kernel_size=1) for C in channels_list
        ])
        C_con = sum(channels_list)
        self.k_proj = nn.Conv1d(C_con, C_con, kernel_size=1)
        self.v_proj = nn.Conv1d(C_con, C_con, kernel_size=1)

        # FFN after PCA
        self.pca_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(C, C, kernel_size=1),
            ) for C in channels_list
        ])

        # 3. PSA: Patch Spatial Attention
        self.qk_proj_psa = nn.Conv1d(C_con, C_con, kernel_size=1)
        self.v_proj_psa = nn.ModuleList([
            nn.Conv1d(C, C, kernel_size=1) for C in channels_list
        ])

        # 4. Recovery: FFN + Reshape
        self.ffn_recovery = nn.ModuleList([
            nn.Sequential(
                nn.Linear(min_P, H_i * W_i),
            ) for (H_i, W_i) in sizes_list
        ])

    def forward(self, *feats):
        """
        Args:
            feats: tuple of tensors
                [f1: (B, 64, H, W),
                 f2: (B, 128, H//2, W//2),
                 f3: (B, 256, H//4, W//4),
                 f4: (B, 512, H//8, W//8),
                 f5: (B, 1024, H//16, W//16)]
        Returns:
            D_list: list of recovered tensors with same shapes
        """
        B = feats[0].shape[0]

        # Step 1: Tokenization → T_list: (B, C_i, min_P)
        T_list = []
        for i, feat in enumerate(feats):
            C_i, H_i, W_i = self.channels_list[i], self.sizes_list[i][0], self.sizes_list[i][1]
            T = feat.view(B, C_i, -1)  # (B, C_i, H_i*W_i)
            T = self.ffn_tokens[i](T)  # (B, C_i, min_P)
            T_list.append(T)

        # Step 2: Patch Channel Attention (PCA)
        Q_list = [self.q_proj[i](T) for i, T in enumerate(T_list)]  # (B, C_i, min_P)

        # K, V from concat(T_norm_list)
        T_concat = torch.cat(T_list, dim=1)  # (B, C_con, min_P)
        K = self.k_proj(T_concat)  # (B, C_con, min_P)
        V = self.v_proj(T_concat)  # (B, C_con, min_P)

        # Compute T_C_i for each Q_i
        T_C_list = []
        for i, Q_i in enumerate(Q_list):
            # Q_i @ K^T -> (B, C_i, C_con)
            attn = torch.matmul(Q_i, K.transpose(-2, -1))  # (B, C_i, C_con)
            attn = nn_functional.softmax(attn, dim=-1)
            # attn @ V^T -> (B, C_i, min_P)
            T_C_i = torch.matmul(attn, V)
            T_C_list.append(T_C_i)

        # FFN + residual
        T_hat_C_list = [
            T_C_list[i] + self.pca_ffn[i](T_C_list[i])
            for i in range(self.num_stages)
        ]

        # Step 3: Patch Spatial Attention (PSA)
        T_hat_C_concat = torch.cat(T_hat_C_list, dim=1)  # (B, C_con, min_P)
        Q_psa = self.qk_proj_psa(T_hat_C_concat)  # (B, C_con, min_P)
        K_psa = self.qk_proj_psa(T_hat_C_concat)
        # Q^T @ K -> (B, min_P, min_P)
        attn_psa = torch.matmul(Q_psa.transpose(-2, -1), K_psa)
        attn_psa = nn_functional.softmax(attn_psa, dim=-1)  # (B, min_P, min_P)

        # V_i for each T_hat_C_i
        V_list = [self.v_proj_psa[i](T_hat_C_list[i]) for i in range(self.num_stages)]  # (B, C_i, min_P)

        # Apply spatial attn: attn_psa @ V_i^T -> (B, min_P, C_i) -> transpose -> (B, C_i, min_P)
        T_S_list = []
        for V_i in V_list:
            T_S_i = torch.matmul(attn_psa, V_i.transpose(-2, -1))  # (B, min_P, C_i)
            T_S_i = T_S_i.transpose(-2, -1)  # (B, C_i, min_P)
            T_S_list.append(T_S_i)

        # Step 4: Patch Recovery → D_i: (B, C_i, H_i, W_i)
        D_list = []
        for i, T_S_i in enumerate(T_S_list):
            C_i, H_i, W_i = self.channels_list[i], self.sizes_list[i][0], self.sizes_list[i][1]
            T_recov = self.ffn_recovery[i](T_S_i)  # (B, C_i, H_i*W_i)
            D_i = T_recov.view(B, C_i, H_i, W_i)
            D_list.append(D_i)

        return D_list  # [D1, D2, D3, D4, D5]