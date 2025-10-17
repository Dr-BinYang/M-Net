import torch
import torch.nn as nn
import torch.nn.functional as nn_functional
import argparse


class DoubleConv(nn.Module):
    """
    Double convolution module: Contains two consecutive Conv2d -> BatchNorm2d -> ReLU operations
    Typically used as the basic unit in U-Net encoders and decoders
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward propagation
        :param x: Input tensor (B, C_in, H, W)
        :return: Output tensor (B, C_out, H, W), dimensions unchanged (due to padding=1)
        """
        return self.conv(x)


class Down(nn.Module):
    """
    Downsampling module: First reduces spatial dimensions (H/2, W/2) through 2x2 max pooling, then applies DoubleConv
    Used in encoders to extract multi-scale features
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.mp_conv = nn.Sequential(
            nn.MaxPool2d(2),                    # Halves spatial dimensions
            DoubleConv(in_channels, out_channels)  # Double convolution for feature extraction
        )

    def forward(self, x):
        """
        Forward propagation
        :param x: Input tensor (B, C_in, H, W)
        :return: Output tensor (B, C_out, H/2, W/2)
        """
        return self.mp_conv(x)


class Up(nn.Module):
    """
    Upsampling module: Uses transposed convolution (ConvTranspose2d) for upsampling, then concatenates with encoder features, followed by DoubleConv for fusion
    Used in decoder skip connection structures
    """
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        # Transposed convolution: Doubles feature map size
        self.up = nn.ConvTranspose2d(
            in_channels,          # Input channels
            in_channels // 2,     # Output channels (typically halved to match skip connection)
            kernel_size=2,
            stride=2
        )
        # After concatenation: in_channels//2 + in_channels//2 = in_channels
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Forward propagation (x1: low-resolution features from upper layer, x2: high-resolution skip connection features from encoder)
        :param x1: Feature map from deep layers (B, C, H, W)
        :param x2: Feature map from corresponding encoder layer (for skip connection)
        :return: Upsampled and fused feature map (B, C_out, 2H, 2W)
        """
        x1 = self.up(x1)  # Transposed convolution upsampling, doubles size

        # Use bilinear interpolation to align x2's spatial dimensions with x1 (avoid mismatches due to odd/even dimensions)
        x2 = nn_functional.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate skip connection features (x2) and upsampled features (x1) along channel dimension
        x_cat = torch.cat([x2, x1], dim=1)  # Concatenated channels = x2.shape[1] + x1.shape[1]

        # Further fuse features through double convolution
        return self.conv(x_cat)


class CUNet(nn.Module):
    """
    CUNet model: Spatio-temporal prediction network based on U-Net architecture
    Input: (B, H, W, F, L)  -> Batch, Height, Width, Input features, Historical timesteps
    Output: (B, H, W, N, T)  -> Batch, Height, Width, Target features, Prediction timesteps
    """
    def __init__(self, args):
        super(CUNet, self).__init__()
        self.args = args
        # Calculate input channels: All input features' historical timesteps merged as channels
        in_channels = len(args.in_features) * args.seq_len
        # Calculate output channels: All target features' prediction timesteps merged as channels
        out_channels = len(args.target_features) * args.pred_len

        # --------------------
        # Encoder (downsampling path)
        # --------------------
        self.inc   = DoubleConv(in_channels, 64)    # Initial convolution layer
        self.down1 = Down(64, 128)                  # First downsampling
        self.down2 = Down(128, 256)                 # Second downsampling
        self.down3 = Down(256, 512)                 # Third downsampling
        self.down4 = Down(512, 1024)                # Fourth downsampling (deepest layer)

        # --------------------
        # Decoder (upsampling path + skip connections)
        # --------------------
        self.up1 = Up(1024, 512)                    # First upsampling
        self.up2 = Up(512, 256)                     # Second upsampling
        self.up3 = Up(256, 128)                     # Third upsampling
        self.up4 = Up(128, 64)                      # Fourth upsampling (restores original resolution)

        # --------------------
        # Output layer
        # --------------------
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)  # 1x1 convolution for final output

    def forward(self, x):
        """
        Forward propagation function
        :param x: Input tensor, shape (B, H, W, F, L)
        :return: Prediction output, shape (B, H, W, N, T)
        """
        B, H, W, F, L = x.shape  # Get input dimension information

        # ========================================================
        # Input preprocessing: Adjust dimension order, merge features and time dimensions as channels
        # ========================================================
        # Original input: (B, H, W, F, L)
        # Adjust to: (B, F, L, H, W) -> then merge F and L dimensions
        x = x.permute(0, 3, 4, 1, 2).contiguous()  # -> (B, F, L, H, W)
        x = x.view(B, F * L, H, W)                 # -> (B, F*L, H, W)

        # ========================================================
        # Encoder: Progressive downsampling to extract multi-scale semantic features
        # ========================================================
        # Shallow features (original resolution)
        encoder_output1 = self.inc(x)               # (B, 64, H, W)
        # Middle features (H/2, W/2)
        encoder_output2 = self.down1(encoder_output1)  # (B, 128, H/2, W/2)
        # Deep features (H/4, W/4)
        encoder_output3 = self.down2(encoder_output2)  # (B, 256, H/4, W/4)
        # Deeper features (H/8, W/8)
        encoder_output4 = self.down3(encoder_output3)  # (B, 512, H/8, W/8)
        # Deepest features (H/16, W/16)
        encoder_output5 = self.down4(encoder_output4)  # (B, 1024, H/16, W/16)

        # ========================================================
        # Decoder: Progressive upsampling with skip connections to recover details
        # ========================================================
        # First upsampling: Fuse deepest and deeper features
        decoder_output1 = self.up1(encoder_output5, encoder_output4)  # (B, 512, H/8, W/8)
        # Second upsampling: Fuse deeper features
        decoder_output2 = self.up2(decoder_output1, encoder_output3)  # (B, 256, H/4, W/4)
        # Third upsampling: Fuse deep features
        decoder_output3 = self.up3(decoder_output2, encoder_output2)  # (B, 128, H/2, W/2)
        # Fourth upsampling: Fuse shallow features, restore original resolution
        decoder_output4 = self.up4(decoder_output3, encoder_output1)  # (B, 64, H, W)

        # ========================================================
        # Output layer: 1x1 convolution generates final prediction
        # ========================================================
        logits = self.outc(decoder_output4)  # (B, N*T, H, W), N: target features, T: prediction steps

        # ========================================================
        # Size alignment: Ensure output spatial dimensions match input (interpolation)
        # ========================================================
        # Interpolate back to original spatial dimensions (H, W)
        # (Theoretically dimensions should match, but odd dimensions may cause deviations)
        x = nn_functional.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        # Now x shape: (B, N*T, H, W)

        # ========================================================
        # Output reconstruction: Restore to original semantic dimension order
        # ========================================================
        # Adjust dimension order: (B, N*T, H, W) -> (B, H, W, N*T)
        x = x.permute(0, 2, 3, 1).contiguous()
        # Split N and T dimensions: (B, H, W, N*T) -> (B, H, W, N, T)
        x = x.view(B, H, W, len(self.args.target_features), self.args.pred_len)

        # ========================================================
        # Return final prediction
        # ========================================================
        return x  # (B, H, W, N, T)


# =====================================================
# ✅ Example usage: Independent test for CUNet
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weather Forecasting Model")

    # Input/output features
    parser.add_argument('--in_features', nargs='+', type=str, default=['d2m', 't2m'], help='Input feature list')
    parser.add_argument('--target_features', nargs='+', type=str, default=['d2m', 't2m'], help='Target feature list')

    # Forecasting task parameters
    parser.add_argument('--model', type=str, default='CUNet', help='Model name')
    parser.add_argument('--seq_len', type=int, default=6, help='Input sequence length (timesteps)')
    parser.add_argument('--pred_len', type=int, default=5, help='Prediction sequence length (timesteps)')
    parser.add_argument('--stride', type=int, default=6, help='Sliding window step size')
    parser.add_argument('--height', type=int, default=209, help='Spatial height')
    parser.add_argument('--width', type=int, default=253, help='Spatial width')

    args = parser.parse_args()

    model = CUNet(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)

    x = torch.randn(4, args.height, args.width, len(args.in_features), args.seq_len).to(device)
    out = model(x)
    print(out.shape)  # Should output torch.Size([4, 209, 253, 2, 5])