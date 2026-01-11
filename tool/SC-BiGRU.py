import torch
import torch.nn as nn
import torch.optim as optim
import math
import gc





class SpatialChunkedBiGRU(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, spatial_chunk_size=48):
        super().__init__()
        self.spatial_chunk_size = spatial_chunk_size
        self.bi_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=False,
            bidirectional=True
        )
        self.output = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        B, T, C, H, W = x.shape
        spatial_chunk = self.spatial_chunk_size
        output = torch.zeros(B, T, self.output.out_features, H, W,
                             device=x.device, dtype=x.dtype)

        num_h = math.ceil(H / spatial_chunk)
        num_w = math.ceil(W / spatial_chunk)

        for h_idx in range(num_h):
            for w_idx in range(num_w):
                h_start = h_idx * spatial_chunk
                h_end = min(h_start + spatial_chunk, H)
                w_start = w_idx * spatial_chunk
                w_end = min(w_start + spatial_chunk, W)

                chunk = x[:, :, :, h_start:h_end, w_start:w_end]
                B_chunk, T_chunk, C_chunk, H_chunk, W_chunk = chunk.shape

                chunk_flat = chunk.permute(0, 3, 4, 1, 2).reshape(B_chunk * H_chunk * W_chunk, T_chunk, C_chunk)
                chunk_transposed = chunk_flat.permute(1, 0, 2)

                gru_out, _ = self.bi_gru(chunk_transposed)
                gru_out = gru_out.permute(1, 0, 2)
                out_flat = self.output(gru_out)

                out_chunk = out_flat.reshape(B_chunk, H_chunk, W_chunk, T_chunk, -1)
                out_chunk = out_chunk.permute(0, 3, 4, 1, 2)
                output[:, :, :, h_start:h_end, w_start:w_end] = out_chunk

        return output

