import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.args = args

        # Input shape: (B, H, W, F, L)
        # We merge F*L as input channels, input becomes (B, F*L, H, W)
        self.input_channels = len(args.in_features) * args.seq_len  # F * L
        self.output_channels = len(args.target_features) * args.pred_len  # N * T

        # Simple convolutional stack (maintains H, W unchanged)
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, self.output_channels, kernel_size=1)  # Output N*T channels
        )

    def forward(self, x):
        """
        x: shape (B, H, W, F, L)
        Returns: (B, H, W, N, T)
        """
        B, H, W, F, L = x.shape

        # Adjust dimensions to (B, F, L, H, W) -> (B, F*L, H, W)
        x = x.permute(0, 3, 4, 1, 2).contiguous()  # -> (B, F, L, H, W)
        x = x.view(B, F * L, H, W)  # -> (B, F*L, H, W)

        # Convolution processing
        x = self.conv(x)  # (B, N*T, H, W)

        # Restore to (B, H, W, N, T)
        N = len(self.args.target_features)
        T = self.args.pred_len
        x = x.view(B, N * T, H, W)
        x = x.permute(0, 2, 3, 1).contiguous()  # -> (B, H, W, N*T)
        x = x.view(B, H, W, N, T)  # -> (B, H, W, N, T)

        return x


# ================ Test logic ================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CNN Model Testing")
    # Input/output features
    parser.add_argument('--in_features', nargs='+', type=str, default=['d2m', 't2m'], help='Input feature list')
    parser.add_argument('--target_features', nargs='+', type=str, default=['d2m', 't2m'], help='Target feature list')

    # Forecasting task parameters
    parser.add_argument('--seq_len', type=int, default=6, help='Input sequence length (timesteps)')
    parser.add_argument('--pred_len', type=int, default=5, help='Prediction sequence length (timesteps)')
    parser.add_argument('--height', type=int, default=209, help='Spatial height')
    parser.add_argument('--width', type=int, default=253, help='Spatial width')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')

    args = parser.parse_args()

    # Initialize model
    model = CNN(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Create test input
    x = torch.randn(args.batch_size, args.height, args.width, len(args.in_features), args.seq_len).to(device)

    # Forward propagation
    out = model(x)
    print(f"[CNN] Output shape: {out.shape}")  # Should output torch.Size([4, 209, 253, 2, 5])