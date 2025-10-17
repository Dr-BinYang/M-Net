import torch
import torch.nn as nn
import torch.nn.functional as nn_functional
import argparse
from tool.attention import *
from tool.DPA import DPA




class DoubleConv(nn.Module):

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
    Downsampling module: First reduces spatial dimensions (H/2, W/2) through 2x2 max pooling, then applies DoubleConv.
    Used in encoders to extract multi-scale features.
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
    Upsampling module: Uses transposed convolution (ConvTranspose2d) for upsampling, then concatenates with encoder features, followed by DoubleConv for fusion.
    Used in decoder skip connection structures.
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
        Forward propagation (x1: low-resolution features from upper layer, x2: high-resolution skip connection features from corresponding encoder layer)
        :param x1: Feature map from deep layers (B, C, H, W)
        :param x2: Feature map from corresponding encoder layer (for skip connection)
        :return: Upsampled and fused feature map (B, C_out, 2H, 2W)
        """
        if x2.shape[2:] != x1.shape[2:]:
            x1 = self.up(x1)  # Transposed convolution upsampling, doubles size
            x1 = nn_functional.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False) # Use bilinear interpolation to align x1's spatial dimensions with x2 (avoid mismatches due to odd/even dimensions)

        # Concatenate skip connection features (x2) and upsampled features (x1) along channel dimension
        x_cat = torch.cat([x2, x1], dim=1)  # Concatenated channels = x2.shape[1] + x1.shape[1]

        # Further fuse features through double convolution
        return self.conv(x_cat)


class DAA(nn.Module):
    """
    Dual Axis Attention module: Channel attention first, then spatial attention.
    Input and output shapes are consistent: (B, C, H, W)
    """

    def __init__(self, in_channels):
        super(DAA, self).__init__()
        self.in_channels = in_channels

        # ------------------------
        # Channel Attention (CA)
        # ------------------------
        # Shared MLP: (B, C, 1, 1) -> (B, C//r, 1, 1) -> (B, C, 1, 1)
        # Uses 1x1 convolution to simulate fully connected layers (more efficient)
        reduction_ratio = 16  # Tunable parameter
        reduced_channels = in_channels // reduction_ratio
        self.ca_mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1)
        )

        # ------------------------
        # Spatial Attention (SA)
        # ------------------------
        # Uses 3x3 convolution to capture larger receptive fields
        self.sa_conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        """
        Forward propagation
        :param x: Input feature map (B, C, H, W)
        :return: Weighted feature map (B, C, H, W)
        """
        B, C, H, W = x.shape

        original_x  = x

        # ========================================================
        # 1. Channel Attention CA: Generates (B, C, 1, 1) weights
        # ========================================================
        avg_pool = nn_functional.adaptive_avg_pool2d(x, (1, 1))  # (B, C, 1, 1)
        max_pool = nn_functional.adaptive_max_pool2d(x, (1, 1))  # (B, C, 1, 1)

        avg_out = self.ca_mlp(avg_pool)  # (B, C, 1, 1)
        max_out = self.ca_mlp(max_pool)  # (B, C, 1, 1)

        channel_out = avg_out + max_out  # (B, C, 1, 1)

        M_C = torch.sigmoid(channel_out)  # (B, C, 1, 1)

        E_prime = x * M_C  # (B, C, H, W) * (B, C, 1, 1) -> broadcast multiplication

        # ========================================================
        # 2. Spatial Attention SA: Generates (B, 1, H, W) weights
        # ========================================================
        avg_pool_s = torch.mean(E_prime, dim=1, keepdim=True)  # (B, 1, H, W)
        max_pool_s, _ = torch.max(E_prime, dim=1, keepdim=True)  # (B, 1, H, W)

        spatial_concat = torch.cat([avg_pool_s, max_pool_s], dim=1)  # (B, 2, H, W)

        M_S = torch.sigmoid(self.sa_conv(spatial_concat))  # (B, 1, H, W)

        E_double_prime = E_prime * M_S  # (B, C, H, W) * (B, 1, H, W) -> broadcast multiplication

        Daa_out = E_double_prime + original_x  # Residual Connection

        return Daa_out  # Daa_out: (B, C, H, W)

class MAF(nn.Module):
    def __init__(self, args,in_channels, out_channels, target_H, target_W, stride):
        super(MAF, self).__init__()
        self.args = args
        self.target_H, self.target_W = target_H, target_W
        kernel_size = 2 * stride
        padding = stride // 2

        self.upsample = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        # Normalization layer
        self.norm = nn.BatchNorm2d(num_features=in_channels)

        # 1x1 convolution layer
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Enable transpose convolution instead of interpolate
        if self.args.use_ConvTranspose2d:
            x = self.upsample(x)

        # Crop or pad to target size
        _, _, h, w = x.shape
        if h != self.target_H or w != self.target_W:
            x = nn_functional.interpolate(x, size=(self.target_H, self.target_W), mode='bilinear', align_corners=False)
        x = self.conv1x1(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


class MNet(nn.Module):
    """
    MNet model: Modified encoder with multi-scale input fusion structure
    """
    def __init__(self, args):
        super(MNet, self).__init__()
        self.args = args
        in_channels = len(args.in_features) * args.seq_len
        self.out_channels = out_channels =  len(args.target_features) * args.pred_len
        self.N = len(args.target_features)
        self.T = args.pred_len
        H, W = args.height, args.width  # Get actual dimensions from args

        # Define different inc modules for each stage
        self.inc0 = DoubleConv(in_channels, 64)      # Initial convolution, output channels: 64
        self.inc1 = DoubleConv(in_channels, 128)     # stage_1 output channels: 128
        self.inc2 = DoubleConv(in_channels, 256)     # stage_2 output channels: 256
        self.inc3 = DoubleConv(in_channels, 512)     # stage_3 output channels: 512
        self.inc4 = DoubleConv(in_channels, 1024)    # stage_4 output channels: 1024

        # Downsampling path (traditional U-Net)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # 1x1 convolution for fusing multi-scale features
        self.fuse1 = nn.Conv2d(128 + 128, 128, kernel_size=1)  # Fuse concatenated results to 128 channels
        self.fuse2 = nn.Conv2d(256 + 256, 256, kernel_size=1)
        self.fuse3 = nn.Conv2d(512 + 512, 512, kernel_size=1)
        self.fuse4 = nn.Conv2d(1024 + 1024, 1024, kernel_size=1)

        # Dynamic branch attention fusion
        self.attn_fusion = DynamicBranchAttention(N=self.N, T=self.T)

        # DAA module for deepest features
        self.daa = DAA(in_channels=1024)

        self.dpa = DPA(
            channels_list=[64, 128, 256, 512, 1024],
            sizes_list=[
                (H, W),  # encoder_output1
                (H // 2, W // 2),  # encoder_output2
                (H // 4, W // 4),  # encoder_output3
                (H // 8, W // 8),  # encoder_output4
                (H // 16, W // 16)  # encoder_output5
            ],
            min_P=1024  # Patch tunable: 256 or 512, controls computation
        )

        # Upsampling path (decoder)
        self.up5 = Up(2048, 1024)
        self.up4 = Up(1024, 512)
        self.up3 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up1 = Up(128, 64)

        # MAF module
        self.maf5 = MAF(args,1024, self.out_channels, H, W, stride=16)  # H/16 -> H
        self.maf4 = MAF(args,512, self.out_channels, H, W, stride=8)  # H/8 -> H
        self.maf3 = MAF(args,256, self.out_channels, H, W, stride=4)  # H/4 -> H
        self.maf2 = MAF(args,128, self.out_channels, H, W, stride=2)   # H/2   -> H (identity)
        self.maf1 = MAF(args,64, self.out_channels, H, W, stride=1)   # H   -> H (identity)

        # Learnable linear transformation layers
        self.proj5 = nn.Conv3d(in_channels=self.N, out_channels=self.N, kernel_size=(3, 3, 3),padding=1)
        self.proj4 = nn.Conv3d(in_channels=self.N, out_channels=self.N, kernel_size=(3, 3, 3),padding=1)
        self.proj3 = nn.Conv3d(in_channels=self.N, out_channels=self.N, kernel_size=(3, 3, 3),padding=1)
        self.proj2 = nn.Conv3d(in_channels=self.N, out_channels=self.N, kernel_size=(3, 3, 3),padding=1)
        self.proj1 = nn.Conv3d(in_channels=self.N, out_channels=self.N, kernel_size=(3, 3, 3),padding=1)


    def forward(self, x):
        B, H, W, F, L = x.shape
        x = x.permute(0, 3, 4, 1, 2).contiguous()  # (B, F, L, H, W)
        y_list=[]

        # ========================================================
        # Encoder: Progressive downsampling to extract multi-scale semantic features
        # ========================================================
        # Stage 0: Original resolution
        x0 = x.view(B, F * L, H, W)                 # -> (B, F*L, H, W)
        encoder_output1 = self.inc0(x0)             # -> (B, 64, H, W)

        # Stage 1: H/2
        x1 = nn_functional.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))  # -> (B, F, L, H/2, W/2)
        x1 = x1.view(B, F * L, H//2, W//2)          # -> (B, F*L, H/2, W/2)
        branch1 = self.inc1(x1)                     # -> (B, 128, H/2, W/2)
        down1_out = self.down1(encoder_output1)     # -> (B, 128, H/2, W/2)
        encoder_output2 = self.fuse1(torch.cat([down1_out, branch1], dim=1))  # -> (B, 128, H/2, W/2)

        # Stage 2: H/4
        x2 = nn_functional.max_pool3d(x, kernel_size=(1, 4, 4), stride=(1, 4, 4))  # -> (B, F, L, H/4, W/4)
        x2 = x2.view(B, F * L, H//4, W//4)          # -> (B, F*L, H/4, W/4)
        branch2 = self.inc2(x2)                     # -> (B, 256, H/4, W/4)
        down2_out = self.down2(encoder_output2)     # -> (B, 256, H/4, W/4)
        encoder_output3 = self.fuse2(torch.cat([down2_out, branch2], dim=1))  # -> (B, 256, H/4, W/4)

        # Stage 3: H/8
        x3 = nn_functional.max_pool3d(x, kernel_size=(1, 8, 8), stride=(1, 8, 8))  # -> (B, F, L, H/8, W/8)
        x3 = x3.view(B, F * L, H//8, W//8)          # -> (B, F*L, H/8, W/8)
        branch3 = self.inc3(x3)                     # -> (B, 512, H/8, W/8)
        down3_out = self.down3(encoder_output3)     # -> (B, 512, H/8, W/8)
        encoder_output4 = self.fuse3(torch.cat([down3_out, branch3], dim=1))  # -> (B, 512, H/8, W/8)

        # Stage 4: H/16
        x4 = nn_functional.max_pool3d(x, kernel_size=(1, 16, 16), stride=(1, 16, 16))  # -> (B, F, L, H/16, W/16)
        x4 = x4.view(B, F * L, H//16, W//16)        # -> (B, F*L, H/16, W/16)
        branch4 = self.inc4(x4)                     # -> (B, 1024, H/16, W/16)
        down4_out = self.down4(encoder_output4)     # -> (B, 1024, H/16, W/16)
        encoder_output5 = self.fuse4(torch.cat([down4_out, branch4], dim=1))  # -> (B, 1024, H/16, W/16)

        # --------------------
        # DAA
        # --------------------
        daa_out = self.daa(encoder_output5.clone())

        # Use DPA to process all encoder outputs (or skip connections)
        if self.args.use_DPA:
            D1, D2, D3, D4, D5 = self.dpa(encoder_output1, encoder_output2, encoder_output3, encoder_output4, encoder_output5)
        else:
            D1, D2, D3, D4, D5 =encoder_output1,encoder_output2,encoder_output3,encoder_output4,encoder_output5

        # --------------------
        # Decoder
        # --------------------
        d5 = self.up5(daa_out, D5)  # -> (B, 1024, H/16, W/16)
        d4 = self.up4(d5, D4)       # -> (B, 512, H/8, W/8)
        d3 = self.up3(d4, D3)       # -> (B, 256, H/4, W/4)
        d2 = self.up2(d3, D2)       # -> (B, 128, H/2, W/2)
        d1 = self.up1(d2, D1)       # -> (B, 64, H, W)

        # MAF prediction
        if self.args.predFusion_Method == 'FinalLayer_One_Fusion':
            out1 = self.maf1(d1).view(B, H, W, self.N, self.T)
            y = out1
            y_list = [out1]
        else:
            out5 = self.maf5(d5).view(B, H, W, self.N, self.T)  # (B, H, W, N, T)
            out4 = self.maf4(d4).view(B, H, W, self.N, self.T)
            out3 = self.maf3(d3).view(B, H, W, self.N, self.T)
            out2 = self.maf2(d2).view(B, H, W, self.N, self.T)
            out1 = self.maf1(d1).view(B, H, W, self.N, self.T)
            if self.args.predFusion_Method=='Linear_All_Fusion' :
                out5 = self.proj5(out5.permute(0, 3, 4, 1, 2)).permute(0, 3, 4, 1, 2)  # (B, H, W, N, T)
                out4 = self.proj4(out4.permute(0, 3, 4, 1, 2)).permute(0, 3, 4, 1, 2)
                out3 = self.proj3(out3.permute(0, 3, 4, 1, 2)).permute(0, 3, 4, 1, 2)
                out2 = self.proj2(out2.permute(0, 3, 4, 1, 2)).permute(0, 3, 4, 1, 2)
                out1 = self.proj1(out1.permute(0, 3, 4, 1, 2)).permute(0, 3, 4, 1, 2)
                y_list = [out5, (out5+out4)/2, (out5+out4+out3)/3, (out5+out4+out3+out2)/4]
                y = (out1 + out2 +  out3 + out4 + out5)/5

            elif self.args.predFusion_Method=='Attention_All_Fusion':
                attn = self.attn_fusion(out1, out2, out3, out4, out5)  # (B, 5)
                y_list = [out1, out2, out3, out4, out5]
                y = (attn[:, 0].view(B, 1, 1, 1, 1) * out1 +
                     attn[:, 1].view(B, 1, 1, 1, 1) * out2 +
                     attn[:, 2].view(B, 1, 1, 1, 1) * out3 +
                     attn[:, 3].view(B, 1, 1, 1, 1) * out4 +
                     attn[:, 4].view(B, 1, 1, 1, 1) * out5)

        return y,y_list



# =====================================================
# ✅ Independent test for MNet
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weather Forecasting Model")
    parser.add_argument('--predFusion_Method', type=str, default='Attention_All_Fusion',
                        help='Final prediction fusion method: FinalLayer_One_Fusion, Attention_All_Fusion, Linear_All_Fusion')

    # Input/output features
    parser.add_argument('--in_features', nargs='+', type=str, default=['d2m', 't2m'], help='Input feature list')
    parser.add_argument('--target_features', nargs='+', type=str, default=['d2m', 't2m'], help='Target feature list')

    # Forecasting task parameters
    parser.add_argument('--model', type=str, default='MNet', help='Model name')
    parser.add_argument('--seq_len', type=int, default=6, help='Input sequence length (timesteps)')
    parser.add_argument('--pred_len', type=int, default=5, help='Prediction sequence length (timesteps)')
    parser.add_argument('--stride', type=int, default=6, help='Sliding window step size')
    parser.add_argument('--height', type=int, default=209, help='Spatial height')
    parser.add_argument('--width', type=int, default=253, help='Spatial width')

    # Model adjustments
    parser.add_argument('--use_DPA', action='store_true', default=False,help='use_DPA')
    parser.add_argument('--use_ConvTranspose2d', action='store_true', default=False,
                        help='Use ConvTranspose2d to restore prediction shape')

    args = parser.parse_args()

    model = MNet(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)

    x = torch.randn(4, args.height, args.width, len(args.in_features), args.seq_len).to(device)

    out,y_list = model(x)

    print(out.shape)  # Should output torch.Size([4, 209, 253, 2, 5])

    # Expected shape
    expected = (4, args.height, args.width, len(args.target_features), args.pred_len)
    is_match = (out.shape == expected)
    print(f"Match: {'Yes' if is_match else 'No'}")
    if not is_match:
        print(f"Expected shape: {expected}")