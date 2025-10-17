from torch.optim import lr_scheduler
from data_provider.data_factory import data_provider
import torch
import torch.nn as nn
from torch import optim
import os
import time
import numpy as np
from colorama import Fore, Style
from tool.plot import *
from collections import defaultdict
from tool.metrics import *


class Exp_Forecast:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model()

        # Print model information before moving to device
        print(f"\n{'=' * 50}")
        print(f"Model architecture: {self.args.model}")
        print(f"{'=' * 50}")
        print(self.model)  # Print model structure

        # Calculate total parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"{'-' * 50}")
        print(f"Total parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"{'=' * 50}\n")

        self.model = self.model.to(self.device)


    def _acquire_device(self):
        if torch.cuda.is_available() and self.args.use_gpu:
            device = torch.device("cuda")
            print(f">> Using GPU")
        else:
            device = torch.device("cpu")  # Default fallback to CPU
            print(f">> Using CPU ")
        return device

    def _build_model(self):
        model_name = self.args.model
        if model_name == 'MNet':
            from model.MNet import MNet
            return MNet(self.args)
        elif model_name == 'CNN':
            from model.CNN import CNN
            return CNN(self.args)
        elif model_name == 'Transformer':
            from model.Transformer import TransformerModel
            return TransformerModel(self.args)
        elif model_name == 'CUNet':
            from model.CUNet import CUNet
            return CUNet(self.args)
        else:
            raise ValueError(f"Model {model_name} not found. "
                             f"Available models: MNet, CNN, Transformer")

    def _get_data(self, flag):
        data_loader,targetsScaler = data_provider(self.args, flag)
        return data_loader,targetsScaler

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def compute_fusion_aware_loss(self,y_final, y_list, y_true, alpha=1.0, beta=0.3):
        """
        Automatically adapt loss behavior based on y_list length and content
        - If y_list contains multiple outputs: multi-level supervision
        - If y_list == [y_final]: only supervise final output (equivalent to single layer)
        """
        criterion = nn.MSELoss()

        # Main loss (final output)
        loss_main = criterion(y_final, y_true)

        # Auxiliary loss: only enabled when y_list has multiple outputs
        if len(y_list) > 1:
            loss_aux = sum(criterion(y_i, y_true) for y_i in y_list) / len(y_list)
        else:
            loss_aux = 0

        total_loss = alpha * loss_main + beta * loss_aux
        return total_loss

    def compute_loss(self, y_final,y_true):
        criterion = nn.MSELoss()
        return criterion(y_final, y_true)

    def train(self):
        train_loader,targetsScaler = self._get_data(flag='train')
        vali_loader,_ = self._get_data(flag='val')
        test_loader,_ = self._get_data(flag='test')

        model_optim = self._select_optimizer()
        mse_criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()

        # Initialize scheduler with parameters from command line
        scheduler = lr_scheduler.ReduceLROnPlateau(
            model_optim,
            mode=self.args.lr_scheduler_mode,
            factor=self.args.lr_scheduler_factor,
            patience=self.args.lr_scheduler_patience,
            verbose=True
        )


        best_vali_loss = float('inf')
        early_stop_counter = 0
        max_patience = self.args.early_stopping_patience  # Early stopping patience
        os.makedirs(self.args.checkpoints, exist_ok=True)
        best_model_path = os.path.join(self.args.checkpoints, 'best_checkpoint.pth')

        # Table header
        print(
            f"{'Epoch':>6} | "
            f"{'Time':>10} | "
            f"{'Train MSE':>10} | {'Train MAE':>10} | {'Train PSNR':>10} |{'Train MS_SSIM':>10} |"
            f"{'Vali MSE':>10} | {'Vali MAE':>10} | {'Vali PSNR':>10} |{'Vali MS_SSIM':>10} | "
            f"{'Test MSE':>10} | {'Test MAE':>10}| {'Test PSNR':>10} |{'Test MS_SSIM':>10}"
        )
        print("-" * 96)

        # Record initial learning rate
        initial_lr = model_optim.param_groups[0]['lr']
        last_lr = initial_lr

        for epoch in range(self.args.train_epochs):
            epoch_time = time.time()
            train_MSE_loss = []
            train_MAE_loss = []
            train_psnr_loss = []
            train_MS_SSIM_loss=[]

            self.model.train()
            for i, (input_tensor, label_tensor, label_time_str, lats_tensor, lons_tensor) in enumerate(train_loader):
                model_optim.zero_grad()
                label_time_str_transposed = [[label_time_str[t][b] for t in range(len(label_time_str))] for b in range(len(label_time_str[0]))]
                input_tensor = input_tensor.float().to(self.device)
                label_tensor = label_tensor.float().to(self.device)

                if self.args.model == 'MNet':
                    outputs, y_list = self.model(input_tensor)
                    train_loss=self.compute_fusion_aware_loss(outputs,y_list, label_tensor)
                else:
                    outputs = self.model(input_tensor)
                    train_loss=self.compute_loss(outputs, label_tensor)

                train_loss.backward()
                model_optim.step()

                # Denormalize inputs and labels
                outputs, label_tensor=targetsScaler.inverse_transform(outputs.cpu().detach()), targetsScaler.inverse_transform(label_tensor.cpu().detach())

                train_MSE_loss.append(mse_criterion(outputs, label_tensor).item())
                train_MAE_loss.append(mae_criterion(outputs, label_tensor).item())
                train_psnr_loss.append(calculate_psnr(outputs, label_tensor).item())
                train_MS_SSIM_loss.append(calculate_ms_ssim(outputs, label_tensor).item())


            # Calculate average losses
            train_MSE = np.average(train_MSE_loss)
            train_MAE = np.average(train_MAE_loss)
            train_psnr = np.average(train_psnr_loss)
            train_MS_SSIM = np.average(train_MS_SSIM_loss)

            # Validation and testing
            vali_MSE, vali_MAE,vali_psnr,vali_MS_SSIM = self.vali(vali_loader,epoch,targetsScaler)
            test_MSE, test_MAE,test_psnr,test_MS_SSIM = self.test(test_loader,epoch,targetsScaler)

            # Call scheduler with validation loss
            scheduler.step(vali_MSE)

            # Manually check if learning rate changed
            new_lr = model_optim.param_groups[0]['lr']
            if abs(new_lr - last_lr) > 1e-12:  # Avoid floating point comparison errors
                print(f"{Fore.YELLOW}→ Epoch {epoch + 1}: Learning rate changed from {last_lr:.2e} to {new_lr:.2e}{Style.RESET_ALL}")
                last_lr = new_lr

            # Check if best model should be saved (based on validation set)
            if vali_MSE < best_vali_loss:
                best_vali_loss = vali_MSE
                torch.save(self.model.state_dict(), best_model_path)
                print("✓ Best model weights obtained (based on validation set)")
                early_stop_counter = 0  # Reset counter
            else:
                early_stop_counter += 1

            # Format time
            elapsed_time = time.time() - epoch_time
            time_str = f"{elapsed_time / 60:.1f} min" if elapsed_time > 60 else f"{elapsed_time:.1f} sec"

            # Print results row
            print(
                f"{Fore.CYAN}{epoch + 1:6d}{Style.RESET_ALL} | "
                f"{Fore.LIGHTBLACK_EX}{time_str:>10}{Style.RESET_ALL} | "
                f"{Fore.GREEN}{train_MSE:10.4f}{Style.RESET_ALL} | {Fore.GREEN}{train_MAE:10.4f}{Style.RESET_ALL} | {Fore.GREEN}{train_psnr:10.4f}{Style.RESET_ALL} | {Fore.GREEN}{train_MS_SSIM:10.4f}{Style.RESET_ALL} |"
                f"{Fore.BLUE}{vali_MSE:10.4f}{Style.RESET_ALL} | {Fore.BLUE}{vali_MAE:10.4f}{Style.RESET_ALL} | {Fore.BLUE}{vali_psnr:10.4f}{Style.RESET_ALL} | {Fore.BLUE}{vali_MS_SSIM:10.4f}{Style.RESET_ALL} | "
                f"{Fore.MAGENTA}{test_MSE:10.4f}{Style.RESET_ALL} | {Fore.MAGENTA}{test_MAE:10.4f}{Style.RESET_ALL} | {Fore.MAGENTA}{test_psnr:10.4f}{Style.RESET_ALL} | {Fore.MAGENTA}{test_MS_SSIM:10.4f}{Style.RESET_ALL}"
            )

            # Early stopping
            if early_stop_counter >= max_patience:
                print(f"\n{Fore.RED}Early stopping triggered at epoch {epoch + 1}.{Style.RESET_ALL}")
                break



        # Load best model
        self.model.load_state_dict(torch.load(best_model_path))
        print("\n✓ Loaded best model weights")
        print("Training complete!!!!!!!!")

    def vali(self, vali_loader,epoch_index,targetsScaler):
        mse_criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()

        self.model.eval()
        vali_MSE_loss = []
        vali_MAE_loss = []
        vali_psnr_loss = []
        vali_MS_SSIM_loss = []

        samples_data = defaultdict(list)  # Store data for first N samples

        with torch.no_grad():
            for i, (input_tensor, label_tensor, label_time_str, lats_tensor, lons_tensor) in enumerate(vali_loader):
                input_tensor = input_tensor.float().to(self.device)
                label_tensor = label_tensor.float().to(self.device)

                if self.args.model == 'MNet':
                    outputs, _ = self.model(input_tensor)
                else:
                    outputs = self.model(input_tensor)

                # Denormalize inputs and labels
                outputs, label_tensor = targetsScaler.inverse_transform(outputs.cpu().detach()), targetsScaler.inverse_transform(label_tensor.cpu().detach())

                vali_MSE_loss.append(mse_criterion(outputs, label_tensor).item())
                vali_MAE_loss.append(mae_criterion(outputs, label_tensor).item())
                vali_psnr_loss.append(calculate_psnr(outputs, label_tensor).item())
                vali_MS_SSIM_loss.append(calculate_ms_ssim(outputs, label_tensor).item())

                if i == 0 and self.args.visualize_vali:
                    batch_size = input_tensor.size(0)
                    N_to_collect = min(self.args.print_N, batch_size)
                    for j in range(N_to_collect):
                        # Directly collect j-th sample (j from 0 to N_to_collect-1)
                        samples_data['lats'].append(lats_tensor[j].cpu().numpy())
                        samples_data['lons'].append(lons_tensor[j].cpu().numpy())
                        samples_data['label_tensor'].append(label_tensor[j].cpu().numpy())
                        samples_data['outputs'].append(outputs[j].cpu().detach().numpy())
                        time_range_for_sample = [label_time_str[t][j] for t in range(len(label_time_str))]
                        samples_data['time_range'].append(time_range_for_sample)
                    first_batch_processed = True  # Mark as processed


        vali_MSE = np.average(vali_MSE_loss)
        vali_MAE = np.average(vali_MAE_loss)
        vali_psnr = np.average(vali_psnr_loss)
        vali_MS_SSIM = np.average(vali_MS_SSIM_loss)


        if first_batch_processed and self.args.visualize_vali:
            plot_multiple_samples_over_time(samples_data, epoch_index=epoch_index, save_dir='./visualizations',stage='vali')

        return vali_MSE, vali_MAE,vali_psnr,vali_MS_SSIM

    def test(self, test_loader,epoch_index,targetsScaler):
        mse_criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()

        self.model.eval()
        test_MSE_loss = []
        test_MAE_loss = []
        test_psnr_loss = []
        test_MS_SSIM_loss = []

        samples_data = defaultdict(list)  # Store data for first N samples

        with torch.no_grad():
            for i, (input_tensor, label_tensor, label_time_str, lats_tensor, lons_tensor) in enumerate(test_loader):
                input_tensor = input_tensor.float().to(self.device)
                label_tensor = label_tensor.float().to(self.device)

                if self.args.model == 'MNet':
                    outputs, _ = self.model(input_tensor)
                else:
                    outputs = self.model(input_tensor)

                # Denormalize inputs and labels
                outputs, label_tensor = targetsScaler.inverse_transform(outputs.cpu().detach()), targetsScaler.inverse_transform(label_tensor.cpu().detach())

                test_MSE_loss.append(mse_criterion(outputs, label_tensor).item())
                test_MAE_loss.append(mae_criterion(outputs, label_tensor).item())
                test_psnr_loss.append(calculate_psnr(outputs, label_tensor).item())
                test_MS_SSIM_loss.append(calculate_ms_ssim(outputs, label_tensor).item())

                if i == 0 and self.args.visualize_Test:
                    batch_size = input_tensor.size(0)
                    N_to_collect = min(self.args.print_N, batch_size)
                    for j in range(N_to_collect):
                        samples_data['lats'].append(lats_tensor[j].cpu().numpy())
                        samples_data['lons'].append(lons_tensor[j].cpu().numpy())
                        samples_data['label_tensor'].append(label_tensor[j].cpu().numpy())
                        samples_data['outputs'].append(outputs[j].cpu().detach().numpy())
                        time_range_for_sample = [label_time_str[t][j] for t in range(len(label_time_str))]
                        samples_data['time_range'].append(time_range_for_sample)
                    first_batch_processed = True  # Mark as processed

        test_MSE = np.average(test_MSE_loss)
        test_MAE = np.average(test_MAE_loss)
        test_psnr = np.average(test_psnr_loss)
        test_MS_SSIM = np.average(test_MS_SSIM_loss)

        if  first_batch_processed and self.args.visualize_Test:
            plot_multiple_samples_over_time(samples_data, epoch_index=epoch_index, save_dir='./visualizations',stage='test')

        return test_MSE, test_MAE,test_psnr,test_MS_SSIM