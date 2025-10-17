import netCDF4 as nc
import torch
from netCDF4 import num2date, date2num
from torch.utils.data import Dataset, DataLoader
import bisect
import os

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from tool.normalization import GlobalFeatureScaler

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)


class Dataset_ERA5(Dataset):
    def __init__(self, args, flag):
        super(Dataset_ERA5, self).__init__()
        self.args = args
        self.flag = flag
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        # Load data
        data = self.load_data(args)

        # Prepare samples
        self.inputs, self.targets, self.lons_sub, self.lats_sub = self.prepare_samples(args, data)

        # Save timestamps
        self.times = data['time'].values[np.argsort(data['time'].values)]  # Sorted timestamps

        # Initialize normalizers
        self.inputScaler = GlobalFeatureScaler(feature_axis=3)
        self.targetsScaler = GlobalFeatureScaler(feature_axis=3)

        # Fit data
        self.inputScaler.fit(self.inputs)
        self.targetsScaler.fit(self.targets)

        # Apply normalization
        self.inputs = self.inputScaler.transform(self.inputs)
        self.targets = self.targetsScaler.transform(self.targets)

        # Access input features and labels corresponding to flag
        self.data_x, self.data_y, self.time_indices = self.__read_data__()
        # Print sample count for current flag
        print(f"[Dataset] {self.flag.capitalize()} set: {len(self.data_x)} samples")

    def load_data(self, args):
        """Load ERA5 data"""
        data_path = os.path.join(args.root_path, args.data_path)
        data = xr.open_dataset(data_path,
                               engine='cfgrib',
                               # backend_kwargs={'indexpath': ''}  # Disable index, can suppress ignore but increases read time
                               )
        self.print_file_info(data, "ERA5 Dataset")
        return data

    def prepare_samples(self, args, data):
        """
        Generate input samples and corresponding ground truth samples for neural network (with integrated latitude/longitude cropping)

        Output:
        Input samples: (B, H, W, F, L)
        Ground truth samples: (B, H, W, N, T)

        Parameters:
        - args.location: Cropping range [south_lat, north_lat, west_lon, east_lon]
        - data: Entire dataset, construct input and ground truth samples using sliding window

        Returns:
        Input samples: (B, H, W, F, L)
        Ground truth samples: (B, H, W, N, T)
        """
        south_lat, north_lat, west_lon, east_lon = args.location
        in_feature_list = args.in_features
        target_feature_list = args.target_features
        seq_len = args.seq_len
        pred_len = args.pred_len
        stride = args.stride if hasattr(args, 'stride') else 1

        # Get time dimension and sort
        time = data['time'].values.copy()
        sort_indices = np.argsort(time)  # Get sorted indices

        # Extract cropped region
        lat_indices, lon_indices, lats_sub, lons_sub = self.get_geo_info(data, south_lat, north_lat, west_lon, east_lon)

        # Rearrange data according to sorted indices
        sorted_data = data.isel(time=sort_indices)

        # Prepare input and target features
        inputs, targets = [], []

        # Process sorted data
        for i in range(0, len(sorted_data['time']) - seq_len - pred_len + 1, stride):
            # Input time window
            in_time_slice = slice(i, i + seq_len)
            # Output time window
            out_time_slice = slice(i + seq_len, i + seq_len + pred_len)

            input_features = []
            for var in in_feature_list:
                feature_data = sorted_data[var].isel(time=in_time_slice, latitude=lat_indices,
                                                     longitude=lon_indices).values
                input_features.append(feature_data)

            target_features = []
            for var in target_feature_list:
                feature_data = sorted_data[var].isel(time=out_time_slice, latitude=lat_indices,longitude=lon_indices).values
                target_features.append(feature_data)

            # (F, L, H, W) -> (H, W, F, L)
            input_features = np.stack(input_features, axis=0)
            input_features = np.transpose(input_features, (2, 3, 0, 1))

            # (N, T, H, W) -> (H, W, N, T)
            target_features = np.stack(target_features, axis=0)
            target_features = np.transpose(target_features, (2, 3, 0, 1))

            inputs.append(input_features)
            targets.append(target_features)

        inputs = np.array(inputs)  # (B, H, W, F, L)
        targets = np.array(targets)  # (B, H, W, N, T)

        return inputs, targets, lons_sub, lats_sub

    def get_geo_info(self, dataset, south_lat, north_lat, west_lon, east_lon):
        """Get latitude/longitude data and cropping indices"""
        # Get latitude/longitude dimensions (supported by both GRIB and NetCDF)
        lats = dataset.latitude.values
        lons = dataset.longitude.values

        # ------------------------- Latitude processing -------------------------
        lat_mask = (lats >= south_lat) & (lats <= north_lat)
        if not lat_mask.any():
            raise ValueError("No points satisfy latitude range")
        lats_sub = lats[lat_mask]
        lat_indices = np.where(lat_mask)[0]

        # ------------------------- Longitude processing -------------------------
        if west_lon <= east_lon:
            lon_mask = (lons >= west_lon) & (lons <= east_lon)
        else:
            lon_mask = (lons >= west_lon) | (lons <= east_lon)
        if not lon_mask.any():
            raise ValueError("No points satisfy longitude range")
        lons_sub = lons[lon_mask]
        lon_indices = np.where(lon_mask)[0]

        return lat_indices, lon_indices, lats_sub, lons_sub

    def print_file_info(self, dataset, name="Dataset"):
        """Print Grid file latitude/longitude range, feature information, and overall data information"""
        print(f"{'=' * 40}")
        print(f"{name} Info:")
        print(f"Dimensions: {dataset.dims}")
        print(f"Variables: {list(dataset.data_vars)}")
        print(f"Time range: {dataset['time'][0].values} to {dataset['time'][-1].values}")
        print(f"Latitude range: {dataset['latitude'][0].values} to {dataset['latitude'][-1].values}")
        print(f"Longitude range: {dataset['longitude'][0].values} to {dataset['longitude'][-1].values}")

        # Calculate spatial resolution
        lats = dataset['latitude'].values
        lons = dataset['longitude'].values
        lat_resolution = np.abs(lats[1] - lats[0]) if len(lats) > 1 else 0
        lon_resolution = np.abs(lons[1] - lons[0]) if len(lons) > 1 else 0

        print(f"Latitude resolution: {lat_resolution:.4f}°")
        print(f"Longitude resolution: {lon_resolution:.4f}°")
        print(f"{'=' * 40}")

    def __read_data__(self):
        num_train = int(len(self.inputs) * 0.7)
        num_test = int(len(self.inputs) * 0.1)
        num_vali = len(self.inputs) - num_train - num_test

        border1s = [0, num_train, num_train + num_vali]  # Start indices
        border2s = [num_train, num_train + num_vali, len(self.inputs)]  # End indices
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        data_x = self.inputs[border1:border2]
        data_y = self.targets[border1:border2]

        # Save start time index for each sample (for timestamp extraction)
        time_indices = np.arange(border1, border2)

        return data_x, data_y, time_indices

    def __len__(self):
        return len(self.data_x) - 1

    def __getitem__(self, idx):
        input_data = self.data_x[idx, :]
        label_data = self.data_y[idx, :]

        # Get timestamp (time corresponding to each label)
        sample_idx = self.time_indices[idx]  # Get original time index
        seq_len = self.args.seq_len
        pred_len = self.args.pred_len
        time_start = sample_idx * self.args.stride  # Start time point
        time_end = time_start + seq_len + pred_len
        time_slice = self.times[time_start:time_end]

        # Only take time part for label
        label_time = time_slice[seq_len:]

        # Convert timestamp to readable string format (for numpy.datetime64)
        label_time_str = [str(pd.to_datetime(t)) for t in label_time]

        # Construct Tensor
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        label_tensor = torch.tensor(label_data, dtype=torch.float32)

        # Latitude/longitude information
        lats_tensor = torch.tensor(self.lats_sub, dtype=torch.float32)
        lons_tensor = torch.tensor(self.lons_sub, dtype=torch.float32)

        return input_tensor, label_tensor, label_time_str, lats_tensor, lons_tensor



def collate_fn(batch):
    inputs, labels, times, lats, lons = zip(*batch)
    inputs = torch.stack(inputs)
    labels = torch.stack(labels)
    lats = torch.stack(lats)
    lons = torch.stack(lons)
    return inputs, labels, times, lats, lons


# ================================================
# ✅ Test if module is runnable
# ================================================
if __name__ == '__main__':
    class Args:
        root_path = 'D:/实验数据集/时空序列数据/ERA5源/'
        data_path = '测试数据.grib'
        location = [3, 55, 73, 136]  # Example cropping range [south_lat, north_lat, west_lon, east_lon:3, 55, 73, 136]
        in_features = ['d2m', 't2m']  # Input features
        target_features = ['d2m', 't2m']  # Target features
        seq_len = 6  # Input length
        pred_len = 3  # Prediction length
        stride = 1  # New sliding step parameter
    args = Args()

    dataset = Dataset_ERA5(args, flag='train')
    print(f"Total samples: {len(dataset)}")

    # Test first sample
    sample_x, sample_y, sample_t, sample_lats, sample_lons = dataset[0]
    print(f"Input shape: {sample_x.shape}")
    print(f"Label shape: {sample_y.shape}")
    print(f"Time stamps: {sample_t}")
    print(f"Lats shape: {sample_lats.shape}")
    print(f"Lons shape: {sample_lons.shape}")

    # Test DataLoader
    # loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for x, y, t, lat, lon in loader:
        print("Batch Input Shape:", x.shape)
        print("Batch Label Shape:", y.shape)
        print("Batch Time Stamps:", t)
        print("Batch Lats Shape:", lat.shape)
        print("Batch Lons Shape:", lon.shape)
        break