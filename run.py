import argparse
import torch
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from main import Exp_Forecast
import argparse

parser = argparse.ArgumentParser(description='Spatio-temporal sequence forecasting training')

# Ablation study
parser.add_argument('--FinalLayer_One_Fusion', type=str, default='Attention_All_Fusion', help='Final prediction fusion method: FinalLayer_One_Fusion, Attention_All_Fusion, Linear_All_Fusion')
parser.add_argument('--use_ConvTranspose2d', action='store_true', default=True, help='Use ConvTranspose2d to restore prediction shape')
parser.add_argument('--use_DPA', action='store_true', default=True, help='Use DPA (middle section)')

# Forecasting task parameters
parser.add_argument('--model', type=str, default='MNet', help='Model name: [MNet, CNN, Transformer, CUNet]')
parser.add_argument('--seq_len', type=int, default=6, help='Input sequence length (timesteps)')
parser.add_argument('--pred_len', type=int, default=5, help='Prediction sequence length (timesteps)')
parser.add_argument('--stride', type=int, default=1, help='Sliding window step size')
parser.add_argument('--d_model', type=int, default=256, help='Hidden dimension per head (d_model)')

# Input/output features
parser.add_argument('--in_features', nargs='+', type=str, default=['d2m', 't2m'], help='Input feature list, e.g.: --in_features d2m t2m')
parser.add_argument('--target_features', nargs='+', type=str, default=['t2m'], help='Target feature list, e.g.: --target_features d2m t2m')

# Optimizer parameters
parser.add_argument('--train_epochs', type=int, default=800, help='Training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--train_criterion', type=str, default='MSE', help='Loss function: MSE, MAE')

# Data path and configuration
#download!!!!!!!!!!!!!!!!: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download
parser.add_argument('--root_path', type=str, default='./dataset/', help='Root directory path for data files')
parser.add_argument('--data_path', type=str, default='ERA5t2m.grib', help='Data file name')
parser.add_argument('--location', nargs=4, type=int, default=[3, 55, 73, 136], help='Cropping range [south_lat north_lat west_lon east_lon], e.g.: --location 3 55 73 136')
parser.add_argument('--height', type=int, default=209, help='Spatial height (latitude)')
parser.add_argument('--width', type=int, default=253, help='Spatial width (longitude)')

# ReduceLROnPlateau parameters
parser.add_argument('--lr_scheduler_mode', type=str, default='min', choices=['min', 'max'], help='Mode for ReduceLROnPlateau scheduler (min or max)')
parser.add_argument('--lr_scheduler_factor', type=float, default=0.5, help='Factor for learning rate reduction in ReduceLROnPlateau')
parser.add_argument('--lr_scheduler_patience', type=int, default=15, help='Patience for ReduceLROnPlateau scheduler')
parser.add_argument('--early_stopping_patience', type=int, default=100, help='Patience for early stopping')

# Visualization control parameters
parser.add_argument('--visualize_Train', action='store_true', default=True, help='Whether to save visualization plots during training')
parser.add_argument('--visualize_vali', action='store_true', default=True, help='Whether to save visualization plots during validation')
parser.add_argument('--visualize_Test', action='store_true', default=True, help='Whether to save visualization plots during testing')
parser.add_argument('--print_N', type=int, default=4, help='Number of validation/test samples to print')
parser.add_argument('--visualize_freq', type=int, default=1, help='Save visualization every N epochs during training/validation')

# GPU settings
parser.add_argument('--use_gpu', action='store_true', default=torch.cuda.is_available(), help='Whether to use GPU (if available)')

# Model weights save path
parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='Path to save model weights')

if __name__ == '__main__':
    args = parser.parse_args()
    print('Args in experiment:')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    exp = Exp_Forecast(args)

    print('>>>>>>>start training <<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train()