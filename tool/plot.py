import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def plot_feature_over_time(lons, lats, label_tensor, outputs,
                          target_feature_idx=0, epoch_index=0,
                          save_dir=None, show=True, stage='train', time_range=None):
    """
    Plot Real/Pred/Bias maps across all timesteps for a specified prediction feature
    :param lons: (253,) Longitude values
    :param lats: (209,) Latitude values
    :param label_tensor: (B, 209, 253, N, T) Ground truth labels
    :param outputs: (B, 209, 253, N, T) Model outputs
    :param target_feature_idx: Index of the feature to visualize
    :param epoch_index: Current epoch number
    :param save_dir: Save directory (optional)
    :param show: Whether to display directly (default True)
    :param stage: Stage name ['train', 'val', 'test'] for subfolder naming
    :param time_range: Time range information (optional)
    """
    # Extract specified feature for the first sample
    B, H, W, N, T = label_tensor.shape
    real_data = label_tensor[0, ..., target_feature_idx, :]  # (209, 253, T)
    pred_data = outputs[0, ..., target_feature_idx, :]       # (209, 253, T)
    bias_data = pred_data - real_data                        # (209, 253, T)

    # Convert to numpy if needed
    real_data = real_data.cpu().numpy() if hasattr(real_data, 'cpu') else real_data
    pred_data = pred_data.cpu().detach().numpy()
    bias_data = bias_data.cpu().detach().numpy()
    lons = lons.cpu().numpy() if hasattr(lons, 'cpu') else lons
    lats = lats.cpu().numpy() if hasattr(lats, 'cpu') else lats

    # =============================
    # Calculate global colorbar ranges
    # =============================
    real_min = real_data.min()
    real_max = real_data.max()
    bias_absmax_global = np.abs(bias_data).max()

    # =============================
    # Build time range information for title
    # =============================
    if time_range is not None and len(time_range) > 0:
        start_time = time_range[0]
        end_time = time_range[-1]
        time_info = f" | Time: {start_time} → {end_time}"
    else:
        time_info = ""


    # Create figure: 3 rows × T columns
    fig = plt.figure(figsize=(4 * T, 12))  # Increase width for colorbar space
    fig.suptitle(f'Epoch {epoch_index} | Feature {target_feature_idx} | Stage: {stage.capitalize()}{time_info}',
                 fontsize=16, y=0.98)

    last_ax1 = None
    last_ax2 = None
    last_ax3 = None

    for t in range(T):
        # === Row 1: Real values ===
        ax1 = fig.add_subplot(3, T, t + 1, projection=ccrs.PlateCarree())
        im1 = ax1.pcolormesh(lons, lats, real_data[..., t],
                             shading='auto', cmap='RdYlBu_r',
                             vmin=real_min, vmax=real_max)
        ax1.add_feature(cfeature.COASTLINE)
        ax1.add_feature(cfeature.LAND, facecolor='lightgray')
        ax1.add_feature(cfeature.OCEAN, facecolor='skyblue')
        gl1 = ax1.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
        gl1.top_labels, gl1.right_labels = False, False
        gl1.xformatter = LONGITUDE_FORMATTER
        gl1.yformatter = LATITUDE_FORMATTER
        ax1.set_title(f"Real (T={t})")

        # === Row 2: Predicted values ===
        ax2 = fig.add_subplot(3, T, T + t + 1, projection=ccrs.PlateCarree())
        im2 = ax2.pcolormesh(lons, lats, pred_data[..., t],
                             shading='auto', cmap='RdYlBu_r',
                             vmin=real_min, vmax=real_max)
        ax2.add_feature(cfeature.COASTLINE)
        ax2.add_feature(cfeature.LAND, facecolor='lightgray')
        ax2.add_feature(cfeature.OCEAN, facecolor='skyblue')
        gl2 = ax2.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
        gl2.top_labels, gl2.right_labels = False, False
        gl2.xformatter = LONGITUDE_FORMATTER
        gl2.yformatter = LATITUDE_FORMATTER
        ax2.set_title(f"Pred (T={t})")

        # === Row 3: Bias ===
        ax3 = fig.add_subplot(3, T, 2*T + t + 1, projection=ccrs.PlateCarree())
        im3 = ax3.pcolormesh(lons, lats, bias_data[..., t],
                             shading='auto', cmap='coolwarm',
                             vmin=-bias_absmax_global, vmax=bias_absmax_global)
        ax3.add_feature(cfeature.COASTLINE)
        ax3.add_feature(cfeature.LAND, facecolor='lightgray')
        ax3.add_feature(cfeature.OCEAN, facecolor='skyblue')
        gl3 = ax3.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
        gl3.top_labels, gl3.right_labels = False, False
        gl3.xformatter = LONGITUDE_FORMATTER
        gl3.yformatter = LATITUDE_FORMATTER
        ax3.set_title(f"Bias (T={t})")

        # Record last column axes for colorbars
        if t == T - 1:
            last_ax1 = ax1
            last_ax2 = ax2
            last_ax3 = ax3

    # =============================
    # Add colorbars (one per row)
    # =============================
    # Reserve space for colorbars (5% width + padding)
    cbar_width = 0.05  # 5% of total width
    cbar_pad = 0.02    # Spacing from main plot

    # Add colorbars using inset_axes
    cax_real = inset_axes(last_ax1, width="5%", height="100%",
                          loc='lower left',
                          bbox_to_anchor=(1 + cbar_pad, 0, 1, 1),
                          bbox_transform=last_ax1.transAxes,
                          borderpad=0)
    fig.colorbar(im1, cax=cax_real, orientation='vertical', label='Real Value')

    cax_pred = inset_axes(last_ax2, width="5%", height="100%",
                          loc='lower left',
                          bbox_to_anchor=(1 + cbar_pad, 0, 1, 1),
                          bbox_transform=last_ax2.transAxes,
                          borderpad=0)
    fig.colorbar(im2, cax=cax_pred, orientation='vertical', label='Pred Value')

    cax_bias = inset_axes(last_ax3, width="5%", height="100%",
                          loc='lower left',
                          bbox_to_anchor=(1 + cbar_pad, 0, 1, 1),
                          bbox_transform=last_ax3.transAxes,
                          borderpad=0)
    fig.colorbar(im3, cax=cax_bias, orientation='vertical', label='Bias')


    # =============================
    # Adjust layout to make space for colorbars
    # =============================
    # left, bottom, right, top, wspace, hspace
    plt.subplots_adjust(
        left=0.05,
        bottom=0.08,
        right=1 - cbar_pad - cbar_width,  # Reserve space for colorbars
        top=0.93,
        wspace=0.3,
        hspace=0.3
    )

    # ============ Save logic ============
    if save_dir is not None:
        # Build full path: save_dir/stage_samples/
        stage_folder = f"{stage}_samples"
        full_save_dir = os.path.join(save_dir, stage_folder)
        os.makedirs(full_save_dir, exist_ok=True)

        plt.savefig(
            os.path.join(full_save_dir, f"epoch_{epoch_index:03d}.png"),
            dpi=150,
            bbox_inches='tight',
            pad_inches=0.5
        )
    if show:
        plt.show()
    else:
        plt.close(fig)



def plot_multiple_samples_over_time(samples_data, epoch_index=0, save_dir=None, stage='vali', target_feature_idx=0, max_samples_per_plot=5):
    """
    Plot time-series feature maps for multiple samples in a single figure.
    Each sample has 2 rows:
        - Row 1: Real (T columns) + Blank (1 column) → Total T+1 columns
        - Row 2: Pred (T columns) + Sum of all Bias (1 column) → Total T+1 columns
    Row 1, column T+1 is blank. Row 2, column T+1 shows accumulated bias.
    Use a main title (suptitle) to display sample information, avoiding misaligned titles.
    """
    N = len(samples_data['lats'])  # Number of samples
    T = samples_data['label_tensor'][0].shape[-1]  # Timesteps

    # Limit maximum displayed samples to prevent oversized canvas
    N = min(N, max_samples_per_plot)

    # Calculate canvas size (2 rows per sample, T+1 columns)
    fig_height = 8 * N  # 2 rows per sample
    fig_width = 4 * (T + 1)  # T+1 columns per sample
    fig = plt.figure(figsize=(fig_width, fig_height))

    # ============ Create main title (suptitle) ============
    suptitle_parts = [f"Epoch {epoch_index} | Feature {target_feature_idx} | Stage: {stage}"]

    sample_info_parts = []
    for n in range(N):
        time_range = samples_data['time_range'][n]
        if time_range is not None and len(time_range) > 0:
            start_time = time_range[0]
            end_time = time_range[-1]
            time_info = f"{start_time}→{end_time}"
        else:
            time_info = "Unknown"
        sample_info_parts.append(f"Sample {n + 1}: {time_info}")

    max_line_length = 80
    current_line = "Samples: "
    all_lines = []

    for info in sample_info_parts:
        if len(current_line) + len(info) + 2 > max_line_length:
            all_lines.append(current_line.rstrip(", "))
            current_line = info + ", "
        else:
            current_line += info + ", "

    if current_line:
        all_lines.append(current_line.rstrip(", "))

    suptitle_text = "\n".join([suptitle_parts[0]] + all_lines)
    fig.suptitle(suptitle_text, fontsize=16, y=0.98, ha='center', va='top')

    # --- Plot subplots ---
    for n in range(N):
        lats = samples_data['lats'][n]
        lons = samples_data['lons'][n]
        real_data = samples_data['label_tensor'][n][..., target_feature_idx, :]  # (H, W, T)
        pred_data = samples_data['outputs'][n][..., target_feature_idx, :]  # (H, W, T)
        bias_data = np.abs(pred_data - real_data)  # (H, W, T)

        # Calculate global color range for current sample
        real_min = real_data.min()
        real_max = real_data.max()
        # Calculate accumulated bias across all timesteps
        bias_sum = np.sum(bias_data, axis=-1)  # (H, W)
        bias_sum_absmax = 50#np.abs(bias_sum).max()    #Need to be unified, preferably constant!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # Calculate starting index for current sample subplots (2 rows, T+1 columns)
        start_idx = n * 2 * (T + 1) + 1

        last_ax_real = None
        last_ax_pred = None
        ax_bias_sum = None  # Specifically record the bias sum plot axis

        # === Row 1: Plot all T real values (Real) ===
        for t in range(T):
            ax1 = fig.add_subplot(2 * N, T + 1, start_idx + t, projection=ccrs.PlateCarree())
            im1 = ax1.pcolormesh(lons, lats, real_data[..., t],
                                 shading='auto', cmap='RdYlBu_r',
                                 vmin=real_min, vmax=real_max)
            ax1.add_feature(cfeature.COASTLINE)
            ax1.add_feature(cfeature.LAND, facecolor='lightgray')
            ax1.add_feature(cfeature.OCEAN, facecolor='skyblue')
            gl1 = ax1.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
            gl1.top_labels, gl1.right_labels = False, False
            gl1.xformatter = LONGITUDE_FORMATTER
            gl1.yformatter = LATITUDE_FORMATTER

            # Control y-axis and x-axis label display
            if t != 0:  # All columns except first
                gl1.left_labels = False
            else:
                gl1.left_labels = True

            gl1.bottom_labels = True  # Show x-axis labels for all columns

            ax1.set_title(f"Real (T={t})")
            if t == T - 1:
                last_ax_real = ax1  # Record last real plot for colorbar

        # === Row 2: Plot all T predicted values (Pred) ===
        for t in range(T):
            row_idx = start_idx + (T + 1) + t  # Start index for row 2 is start_idx + (T+1)
            ax2 = fig.add_subplot(2 * N, T + 1, row_idx, projection=ccrs.PlateCarree())
            im2 = ax2.pcolormesh(lons, lats, pred_data[..., t],
                                 shading='auto', cmap='RdYlBu_r',
                                 vmin=real_min, vmax=real_max)
            ax2.add_feature(cfeature.COASTLINE)
            ax2.add_feature(cfeature.LAND, facecolor='lightgray')
            ax2.add_feature(cfeature.OCEAN, facecolor='skyblue')
            gl2 = ax2.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
            gl2.top_labels, gl2.right_labels = False, False
            gl2.xformatter = LONGITUDE_FORMATTER
            gl2.yformatter = LATITUDE_FORMATTER

            # Control y-axis and x-axis label display
            if t != 0:  # All columns except first
                gl2.left_labels = False
            else:
                gl2.left_labels = True

            gl2.bottom_labels = True  # Show x-axis labels for all columns

            ax2.set_title(f"Pred (T={t})")
            if t == T - 1:
                last_ax_pred = ax2  # Record last pred plot for colorbar

        # === Extra: Plot accumulated bias (Bias Sum) ===
        bias_sum_col_idx = start_idx + (T + 1) + T  # Position T+1 in row 2
        ax3 = fig.add_subplot(2 * N, T + 1, bias_sum_col_idx, projection=ccrs.PlateCarree())
        im3 = ax3.pcolormesh(lons, lats, bias_sum,
                             shading='auto', cmap='jet',
                             vmin=0, vmax=bias_sum_absmax)
        ax3.add_feature(cfeature.COASTLINE)
        ax3.add_feature(cfeature.LAND, facecolor='lightgray')
        ax3.add_feature(cfeature.OCEAN, facecolor='skyblue')
        gl3 = ax3.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
        gl3.top_labels, gl3.right_labels = False, False
        gl3.xformatter = LONGITUDE_FORMATTER
        gl3.yformatter = LATITUDE_FORMATTER

        # Control y-axis and x-axis label display
        if T != 0:  # Hide y-axis labels if not first column
            gl3.left_labels = False
        else:
            gl3.left_labels = True

        gl3.bottom_labels = True  # Show x-axis labels

        ax3.set_title(f"Bias Sum\n(T=0 to T={T - 1})")
        ax_bias_sum = ax3


        # Add colorbar for current sample
        # Real colorbar
        if last_ax_real:
            cbar_pad = 0.02
            cax_real = inset_axes(last_ax_real, width="5%", height="100%",
                                  loc='lower left',
                                  bbox_to_anchor=(1  + cbar_pad, 0, 1, 1),
                                  bbox_transform=last_ax_real.transAxes,
                                  borderpad=0)
            fig.colorbar(im1, cax=cax_real, orientation='vertical')

        # Pred colorbar
        if last_ax_pred:
            cbar_pad = 0.02
            cax_pred = inset_axes(last_ax_pred, width="5%", height="100%",
                                  loc='lower left',
                                  bbox_to_anchor=(1  + cbar_pad, 0, 1, 1),
                                  bbox_transform=last_ax_pred.transAxes,
                                  borderpad=0)
            fig.colorbar(im2, cax=cax_pred, orientation='vertical')

        # Bias Sum colorbar
        if ax_bias_sum:
            cbar_pad = 0.02
            cax_bias_sum = inset_axes(ax_bias_sum, width="5%", height="100%",
                                      loc='lower left',
                                      bbox_to_anchor=(1  + cbar_pad, 0, 1, 1),
                                      bbox_transform=ax_bias_sum.transAxes,
                                      borderpad=0)
            fig.colorbar(im3, cax=cax_bias_sum, orientation='vertical')

    # ============ Adjust overall layout ============
    plt.subplots_adjust(
        left=0.05,
        bottom=0.08,
        right=0.8,  # Reserve space for colorbars
        top=0.93,
        wspace=0.2,
        hspace=0.01
    )

    # ============ Save logic ============
    if save_dir is not None:
        stage_folder = f"{stage}_samples"
        full_save_dir = os.path.join(save_dir, stage_folder)
        os.makedirs(full_save_dir, exist_ok=True)

        plt.savefig(
            os.path.join(full_save_dir, f"epoch_{epoch_index:03d}.png"),
            dpi=150,
            bbox_inches='tight',
            pad_inches=0.5
        )

    plt.close(fig)


def plot_multiple_samples_over_time_3row(samples_data, epoch_index=0, save_dir=None, stage='vali', target_feature_idx=0, max_samples_per_plot=5):
    """
    Plot time-series feature maps for multiple samples in a single figure.
    Each sample has 3 rows (Real, Pred, Bias), each with T columns (timesteps).
    Use a main title (suptitle) to display sample information, avoiding misaligned titles.
    """
    N = len(samples_data['lats'])  # Number of samples
    T = samples_data['label_tensor'][0].shape[-1]  # Timesteps

    # Limit maximum displayed samples to prevent oversized canvas
    N = min(N, max_samples_per_plot)

    # Calculate canvas size
    fig_height = 12 * N  # 3 rows per sample, ~4 units height per row
    fig_width = 4 * T
    fig = plt.figure(figsize=(fig_width, fig_height))

    # ============ Create main title (suptitle) ============
    # Build main title containing all sample information
    suptitle_parts = [f"Epoch {epoch_index} | Feature {target_feature_idx} | Stage: {stage}"]

    sample_info_parts = []
    for n in range(N):
        time_range = samples_data['time_range'][n]
        if time_range is not None and len(time_range) > 0:
            start_time = time_range[0]
            end_time = time_range[-1]
            time_info = f"{start_time}→{end_time}"
        else:
            time_info = "Unknown"
        sample_info_parts.append(f"Sample {n + 1}: {time_info}")

    # Combine sample info into one or more lines
    # May need line breaks if too many samples or long time strings
    max_line_length = 80  # Assume max 80 characters per line
    current_line = "Samples: "
    all_lines = []

    for info in sample_info_parts:
        if len(current_line) + len(info) + 2 > max_line_length:  # +2 for ", "
            all_lines.append(current_line.rstrip(", "))
            current_line = info + ", "
        else:
            current_line += info + ", "

    # Add last line
    if current_line:
        all_lines.append(current_line.rstrip(", "))

    # Connect all lines with newline
    suptitle_text = "\n".join([suptitle_parts[0]] + all_lines)

    # Add main title
    fig.suptitle(suptitle_text, fontsize=16, y=0.98, ha='center', va='top')

    # --- Plot subplots ---
    for n in range(N):  # For each sample
        lats = samples_data['lats'][n]
        lons = samples_data['lons'][n]
        # Extract data for specified feature (target_feature_idx)
        real_data = samples_data['label_tensor'][n][..., target_feature_idx, :]  # (H, W, T)
        pred_data = samples_data['outputs'][n][..., target_feature_idx, :]  # (H, W, T)
        bias_data = pred_data - real_data  # (H, W, T)

        # Calculate global color range for current sample
        real_min = real_data.min()
        real_max = real_data.max()
        bias_absmax_global = np.abs(bias_data).max()

        # Calculate starting index for current sample subplots
        # First subplot for nth sample (n from 0) is n * 3 * T + 1
        start_idx = n * 3 * T + 1

        last_ax1 = None
        last_ax2 = None
        last_ax3 = None

        for t in range(T):
            # === Row 1: Real values (Real) ===
            ax1 = fig.add_subplot(3 * N, T, start_idx + t, projection=ccrs.PlateCarree())
            im1 = ax1.pcolormesh(lons, lats, real_data[..., t],
                                 shading='auto', cmap='RdYlBu_r',
                                 vmin=real_min, vmax=real_max)
            ax1.add_feature(cfeature.COASTLINE)
            ax1.add_feature(cfeature.LAND, facecolor='lightgray')
            ax1.add_feature(cfeature.OCEAN, facecolor='skyblue')
            gl1 = ax1.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
            gl1.top_labels, gl1.right_labels = False, False
            gl1.xformatter = LONGITUDE_FORMATTER
            gl1.yformatter = LATITUDE_FORMATTER
            ax1.set_title(f"Real (T={t})")

            # === Row 2: Predicted values (Pred) ===
            ax2 = fig.add_subplot(3 * N, T, start_idx + T + t, projection=ccrs.PlateCarree())
            im2 = ax2.pcolormesh(lons, lats, pred_data[..., t],
                                 shading='auto', cmap='RdYlBu_r',
                                 vmin=real_min, vmax=real_max)
            ax2.add_feature(cfeature.COASTLINE)
            ax2.add_feature(cfeature.LAND, facecolor='lightgray')
            ax2.add_feature(cfeature.OCEAN, facecolor='skyblue')
            gl2 = ax2.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
            gl2.top_labels, gl2.right_labels = False, False
            gl2.xformatter = LONGITUDE_FORMATTER
            gl2.yformatter = LATITUDE_FORMATTER
            ax2.set_title(f"Pred (T={t})")

            # === Row 3: Bias ===
            ax3 = fig.add_subplot(3 * N, T, start_idx + 2 * T + t, projection=ccrs.PlateCarree())
            im3 = ax3.pcolormesh(lons, lats, bias_data[..., t],
                                 shading='auto', cmap='coolwarm',
                                 vmin=-bias_absmax_global, vmax=bias_absmax_global)
            ax3.add_feature(cfeature.COASTLINE)
            ax3.add_feature(cfeature.LAND, facecolor='lightgray')
            ax3.add_feature(cfeature.OCEAN, facecolor='skyblue')
            gl3 = ax3.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
            gl3.top_labels, gl3.right_labels = False, False
            gl3.xformatter = LONGITUDE_FORMATTER
            gl3.yformatter = LATITUDE_FORMATTER
            ax3.set_title(f"Bias (T={t})")

            # Record last column axes for colorbars
            if t == T - 1:
                last_ax1 = ax1
                last_ax2 = ax2
                last_ax3 = ax3

        # Add colorbars for current sample's three rows
        if last_ax1 and last_ax2 and last_ax3:
            cbar_pad = 0.02
            cax_real = inset_axes(last_ax1, width="5%", height="100%",
                                  loc='lower left',
                                  bbox_to_anchor=(1 + cbar_pad, 0, 1, 1),
                                  bbox_transform=last_ax1.transAxes,
                                  borderpad=0)
            fig.colorbar(im1, cax=cax_real, orientation='vertical', label='Value')

            cax_pred = inset_axes(last_ax2, width="5%", height="100%",
                                  loc='lower left',
                                  bbox_to_anchor=(1 + cbar_pad, 0, 1, 1),
                                  bbox_transform=last_ax2.transAxes,
                                  borderpad=0)
            fig.colorbar(im2, cax=cax_pred, orientation='vertical', label='Value')

            cax_bias = inset_axes(last_ax3, width="5%", height="100%",
                                  loc='lower left',
                                  bbox_to_anchor=(1 + cbar_pad, 0, 1, 1),
                                  bbox_transform=last_ax3.transAxes,
                                  borderpad=0)
            fig.colorbar(im3, cax=cax_bias, orientation='vertical', label='Bias')

    # ============ Adjust overall layout ============
    top_margin = 0.98  # y-position for suptitle
    # Calculate space needed for suptitle
    # Suptitle space ≈ font size * number of lines * 0.02 (relative to total height)
    suptitle_space = (16 * (len(all_lines) + 1)) / (fig.dpi * fig_height)
    # Or simply set a reasonable top value
    plt.subplots_adjust(
        left=0.05,
        bottom=0.08,
        right=0.9,
        top=0.93,  # Reserve space for suptitle, may need adjustment based on N and lines
        wspace=0.3,
        hspace=0.4
    )

    # ============ Save logic ============
    if save_dir is not None:
        stage_folder = f"{stage}_samples"
        full_save_dir = os.path.join(save_dir, stage_folder)
        os.makedirs(full_save_dir, exist_ok=True)

        plt.savefig(
            os.path.join(full_save_dir, f"epoch_{epoch_index:03d}.png"),
            dpi=150,
            bbox_inches='tight',
            pad_inches=0.5
        )

    plt.close(fig)