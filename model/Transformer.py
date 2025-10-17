import torch
import torch.nn as nn
import argparse



class SinusoidalPositionEmbedding2D(nn.Module):
    """2D Sinusoidal Positional Encoding"""

    def __init__(self, height, width, d_model):
        super(SinusoidalPositionEmbedding2D, self).__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2D pos encoding.")

        d_half = d_model // 2
        pe = torch.zeros(1, height, width, d_model)

        # Generate all row and column indices (H, 1) and (1, W)
        row = torch.arange(0, height).unsqueeze(1)  # (H, 1)
        col = torch.arange(0, width).unsqueeze(1)  # (W, 1)

        # Generate frequency terms: sin(10000^{-2i/d}), step size 2
        div_term = torch.exp(torch.arange(0, d_half, 2).float() * (-torch.log(torch.tensor(10000.0)) / (d_half - 1)))
        div_term = div_term.unsqueeze(0)  # (1, d_half//2)

        # Calculate row encoding: sin(row * div_term), cos(row * div_term)
        # Result: (H, 1) @ (1, d_half//2) -> (H, d_half//2) -> expand to (H, 1, d_half//2)
        sin_row = torch.sin(row * div_term)  # (H, d_half//2)
        cos_row = torch.cos(row * div_term)
        row_emb = torch.stack([sin_row, cos_row], dim=-1).reshape(height, -1)  # (H, d_half)

        # Calculate column encoding
        sin_col = torch.sin(col * div_term)  # (W, d_half//2)
        cos_col = torch.cos(col * div_term)
        col_emb = torch.stack([sin_col, cos_col], dim=-1).reshape(width, -1)  # (W, d_half)

        # Expand row_emb to (H, W, d_half), col_emb to (H, W, d_half)
        row_emb = row_emb.unsqueeze(1).repeat(1, width, 1)  # (H, W, d_half)
        col_emb = col_emb.unsqueeze(0).repeat(height, 1, 1)  # (H, W, d_half)

        # Concatenate to form complete encoding
        pe[0] = torch.cat([row_emb, col_emb], dim=-1)  # (H, W, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, H, W, d_model)
        return x + self.pe  # Automatically broadcast B dimension


class TransformerModel(nn.Module):
    def __init__(self, args):
        super(TransformerModel, self).__init__()
        self.args = args

        # Use height and width defined in args
        self.H = args.height  # Spatial height
        self.W = args.width  # Spatial width
        self.F = len(args.in_features)  # Number of input features
        self.L = args.seq_len  # History length
        self.N = len(args.target_features)  # Number of output features
        self.T = args.pred_len  # Prediction length

        input_dim = self.F * self.L
        output_dim = self.N * self.T
        d_model = args.d_model  # Get d_model value from args
        nhead = 8
        num_layers = 3

        self.d_model = d_model

        # Input projection layer
        self.input_proj = nn.Linear(input_dim, d_model)

        # Build positional encoding using H, W provided by args
        self.pos_encoder = SinusoidalPositionEmbedding2D(self.H, self.W, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection layer
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

        # LayerNorm layer
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: shape (B, H, W, F, L)
        return: (B, H, W, N, T)
        """
        B, H, W, F, L = x.shape

        # Check if input dimensions match model configuration
        if H != self.H or W != self.W:
            raise ValueError(f"Input spatial size ({H}, {W}) does not match model config ({self.H}, {self.W}).")

        # Merge F and L dimensions
        x = x.view(B, H, W, -1)  # (B, H, W, F*L)
        x = self.input_proj(x)  # -> (B, H, W, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)  # (B, H, W, d_model)

        # Flatten into spatial sequence
        x = x.view(B, H * W, self.d_model)  # (B, S, D)
        x = self.norm(x)

        # Transformer encoding
        x = self.transformer(x)  # (B, S, D)

        # Restore spatial structure
        x = x.view(B, H, W, self.d_model)

        # Output mapping
        x = self.output_proj(x)  # (B, H, W, N*T)
        x = x.view(B, H, W, self.N, self.T)  # (B, H, W, N, T)

        return x


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weather Forecasting Model")

    # Input/output features
    parser.add_argument('--in_features', nargs='+', type=str, default=['d2m', 't2m'], help='Input feature list')
    parser.add_argument('--target_features', nargs='+', type=str, default=['d2m', 't2m'], help='Target feature list')

    # Forecasting task parameters
    parser.add_argument('--model', type=str, default='Transformer', help='Model name')
    parser.add_argument('--seq_len', type=int, default=6, help='Input sequence length (timesteps)')
    parser.add_argument('--pred_len', type=int, default=5, help='Prediction sequence length (timesteps)')
    parser.add_argument('--stride', type=int, default=6, help='Sliding window step size')
    parser.add_argument('--height', type=int, default=209, help='Spatial height')
    parser.add_argument('--width', type=int, default=253, help='Spatial width')
    parser.add_argument('--d_model', type=int, default=128, help='Transformer hidden dimension d_model')  # Modified to smaller value like 128

    args = parser.parse_args()

    # Initialize model
    model = TransformerModel(args)

    # Use CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move model to specified device
    model = model.to(device)

    # Test input
    x = torch.randn(4, args.height, args.width, len(args.in_features), args.seq_len).to(device)
    out = model(x)
    print(out.shape)  # Should output torch.Size([4, 209, 253, 2, 5])