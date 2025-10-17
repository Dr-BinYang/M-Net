import torch
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.functional import multiscale_structural_similarity_index_measure


def calculate_psnr(pred, target):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)

    Parameters:
        pred: Prediction tensor, shape (batch, height, width, channels, time)
        target: Target tensor, shape (batch, height, width, channels, time)

    Returns:
        PSNR value (scalar)
    """
    # Reshape to format suitable for image metrics: (batch*time, channels, height, width)
    pred_reshaped = pred.permute(0, 4, 3, 1, 2).contiguous().view(-1, pred.shape[3], pred.shape[1], pred.shape[2])
    target_reshaped = target.permute(0, 4, 3, 1, 2).contiguous().view(-1, target.shape[3], target.shape[1],
                                                                      target.shape[2])

    # Calculate data range
    data_range = float(target.max() - target.min())
    if data_range == 0:  # Avoid division by zero
        data_range = 1.0

    # Calculate PSNR
    psnr_metric = PeakSignalNoiseRatio(data_range=data_range)
    return psnr_metric(pred_reshaped, target_reshaped)


def calculate_ms_ssim(pred, target):
    """
    Calculate Multi-Scale Structural Similarity Index (MS-SSIM)

    Parameters:
        pred: Prediction tensor, shape (batch, height, width, channels, time)
        target: Target tensor, shape (batch, height, width, channels, time)

    Returns:
        MS-SSIM value (scalar)
    """
    # Reshape to format suitable for image metrics: (batch*time, channels, height, width)
    pred_reshaped = pred.permute(0, 4, 3, 1, 2).contiguous().view(-1, pred.shape[3], pred.shape[1], pred.shape[2])
    target_reshaped = target.permute(0, 4, 3, 1, 2).contiguous().view(-1, target.shape[3], target.shape[1],
                                                                      target.shape[2])

    # Calculate data range
    data_range = float(target.max() - target.min())
    if data_range == 0:  # Avoid division by zero
        data_range = 1.0

    # Calculate MS-SSIM
    kernel_size = min(11, min(pred_reshaped.shape[2], pred_reshaped.shape[3]))
    return multiscale_structural_similarity_index_measure(
        pred_reshaped,
        target_reshaped,
        data_range=data_range,
        kernel_size=kernel_size
    )