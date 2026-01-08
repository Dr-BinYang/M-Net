import torch
import torch.nn as nn
import torch.optim as optim
import math
import gc
import traceback


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


class SpatialChunkedBiGRU(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, spatial_chunk_size=48):
        super().__init__()
        self.spatial_chunk_size = spatial_chunk_size
        self.bi_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.output = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # Input shape: [B, H, W, T, C]
        B, H, W, T, C = x.shape
        spatial_chunk = self.spatial_chunk_size
        # Output shape: [B, H, W, T, output_dim]
        output = torch.zeros(B, H, W, T, self.output.out_features,
                             device=x.device, dtype=x.dtype)

        num_h = math.ceil(H / spatial_chunk)
        num_w = math.ceil(W / spatial_chunk)

        for h_idx in range(num_h):
            for w_idx in range(num_w):
                h_start = h_idx * spatial_chunk
                h_end = min(h_start + spatial_chunk, H)
                w_start = w_idx * spatial_chunk
                w_end = min(w_start + spatial_chunk, W)

                chunk = x[:, h_start:h_end, w_start:w_end, :, :]
                B_chunk, H_chunk, W_chunk, T_chunk, C_chunk = chunk.shape

                chunk_flat = chunk.reshape(B_chunk * H_chunk * W_chunk, T_chunk, C_chunk)

                gru_out, _ = self.bi_gru(chunk_flat)  # Output shape: [B*H*W, T, hidden_dim*2]

                out_flat = self.output(gru_out)  # Shape: [B*H*W, T, output_dim]

                out_chunk = out_flat.reshape(B_chunk, H_chunk, W_chunk, T_chunk, -1)
                output[:, h_start:h_end, w_start:w_end, :, :] = out_chunk

        return output


def safe_operation(operation_func, operation_name):
    try:
        operation_func()
        success = True
        error_msg = None
    except Exception as e:
        success = False
        error_msg = str(e)
        print(f"  {operation_name} error: {str(e)[:100]}")
        if "CUDA out of memory" in str(e):
            print(f"  CUDA memory insufficient, recording current peak memory")

    peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
    return success, peak_memory, error_msg


def test_full_training_cycle():
    print("=" * 60)
    print("Complete Training Cycle Peak Memory Test")
    print("=" * 60)

    clear_memory()

    if not torch.cuda.is_available():
        print("Error: No GPU detected!")
        return

    device = torch.device('cuda')

    B, H, W, T, C = 1, 384, 384, 12, 1
    hidden_dim = 64

    print(f"Configuration:")
    print(f"  Input shape: [{B}, {H}, {W}, {T}, {C}]")
    print(f"  Total pixels: {B * H * W:,}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  GPU model: {torch.cuda.get_device_name(0)}")
    print(f"  GPU total memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    chunk_sizes = [8, 128, 384]
    training_results = []

    for chunk_size in chunk_sizes:
        print(f"\n" + "-" * 40)
        print(f"Testing chunk size: {chunk_size}×{chunk_size}")
        print(f"Pixels per chunk: {B * chunk_size * chunk_size:,}")
        print(
            f"Number of chunks: {math.ceil(H / chunk_size)}×{math.ceil(W / chunk_size)} = {math.ceil(H / chunk_size) * math.ceil(W / chunk_size)}")

        clear_memory()

        result = {
            'chunk_size': chunk_size,
            'forward_peak': 0,
            'forward_success': False,
            'training_peak': 0,
            'training_success': False,
            'avg_peak': 0,
            'max_peak': 0,
            'stability_test_success': False,
            'error_msgs': []
        }

        print("\n1. Forward pass only test:")

        try:
            model = SpatialChunkedBiGRU(
                input_dim=C,
                hidden_dim=hidden_dim,
                output_dim=1,
                spatial_chunk_size=chunk_size
            ).to(device)
            model.eval()

            x = torch.randn(B, H, W, T, C, device=device)

            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                output = model(x)

            forward_peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"  Forward peak memory: {forward_peak:.2f} GB")

            result['forward_peak'] = forward_peak
            result['forward_success'] = True

            del output

        except Exception as e:
            forward_peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"  Forward pass error, peak memory: {forward_peak:.2f} GB")
            print(f"  Error: {str(e)[:100]}")

            result['forward_peak'] = forward_peak
            result['forward_success'] = False
            result['error_msgs'].append(f"Forward pass: {str(e)[:100]}")

            if 'model' in locals():
                del model
            if 'x' in locals():
                del x

        print("\n2. Complete training step test:")

        clear_memory()
        torch.cuda.reset_peak_memory_stats()

        try:
            model = SpatialChunkedBiGRU(
                input_dim=C,
                hidden_dim=hidden_dim,
                output_dim=1,
                spatial_chunk_size=chunk_size
            ).to(device)
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            # Generate new data - new shape
            x = torch.randn(B, H, W, T, C, device=device)
            target = torch.randn(B, H, W, T, 1, device=device)

            # Complete training step
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            training_peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"  Training peak memory: {training_peak:.2f} GB")
            print(f"  Training loss: {loss.item():.6f}")

            result['training_peak'] = training_peak
            result['training_success'] = True

            print("\n3. Multiple iteration stability test:")

            clear_memory()
            torch.cuda.reset_peak_memory_stats()

            peak_history = []
            success_iterations = 0
            for i in range(5):
                try:
                    optimizer.zero_grad()
                    output = model(x)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    current_peak = torch.cuda.max_memory_allocated() / 1024 ** 3
                    peak_history.append(current_peak)
                    success_iterations += 1

                    x = x * 0.99 + torch.randn_like(x) * 0.01

                except Exception as e:
                    current_peak = torch.cuda.max_memory_allocated() / 1024 ** 3
                    peak_history.append(current_peak)
                    print(f"  Iteration {i + 1} error, current peak memory: {current_peak:.2f} GB")
                    break

            if peak_history:
                avg_peak = sum(peak_history) / len(peak_history)
                max_peak = max(peak_history)
                print(f"  Average peak memory: {avg_peak:.2f} GB")
                print(f"  Maximum peak memory: {max_peak:.2f} GB")
                print(f"  Successful iterations: {success_iterations}/5")

                result['avg_peak'] = avg_peak
                result['max_peak'] = max_peak
                result['stability_test_success'] = (success_iterations > 0)

            del model, optimizer, criterion, x, target, output, loss

        except Exception as e:
            training_peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"  Training step error, peak memory: {training_peak:.2f} GB")
            print(f"  Error: {str(e)[:100]}")

            result['training_peak'] = training_peak
            result['training_success'] = False
            result['error_msgs'].append(f"Training step: {str(e)[:100]}")

            if 'model' in locals():
                del model
            if 'x' in locals():
                del x

        finally:
            clear_memory()

        result['success'] = result['forward_success'] and result['training_success'] and result[
            'stability_test_success']

        training_results.append(result)

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU total memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    print(f"Input resolution: {H}×{W}")
    print(f"Batch size: {B}")
    print(f"Time steps: {T}")
    print(f"Hidden dimension: {hidden_dim}")

    print(f"\nMemory Consumption Comparison:")
    print(
        f"{'Chunk Size':<10} | {'Forward Mem(GB)':<12} | {'Training Peak(GB)':<12} | {'Max Peak(GB)':<12} | {'Status':<10}")
    print("-" * 70)

    for result in training_results:
        chunk_str = f"{result['chunk_size']}×{result['chunk_size']}"

        forward_mem = f"{result['forward_peak']:.2f}" if result['forward_peak'] > 0 else "N/A"

        training_mem = f"{result['training_peak']:.2f}" if result['training_peak'] > 0 else "N/A"

        max_mem = f"{result['max_peak']:.2f}" if result['max_peak'] > 0 else "N/A"

        if result['success']:
            status = "✓ Pass"
        elif result['forward_success'] and not result['training_success']:
            status = "✗ Training failed"
        elif not result['forward_success']:
            status = "✗ Forward failed"
        else:
            status = "✗ Unstable"

        print(f"{chunk_str:>10} | "
              f"{forward_mem:>12} | "
              f"{training_mem:>12} | "
              f"{max_mem:>12} | "
              f"{status:<10}")

        if result['error_msgs']:
            print(f"{' ':>10}   Errors: {', '.join(result['error_msgs'])}")

    print(f"\nFeasibility Analysis:")
    for result in training_results:
        if result['max_peak'] > 0:  # Analyze if there's any memory record
            gpu_capacity = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            usage_percentage = result['max_peak'] / gpu_capacity * 100 if result['max_peak'] > 0 else 0

            if result['success']:
                if result['max_peak'] < gpu_capacity * 0.8:
                    feasibility = f"Fully feasible (uses {usage_percentage:.1f}%)"
                elif result['max_peak'] < gpu_capacity:
                    feasibility = f"Barely feasible (uses {usage_percentage:.1f}%)"
                else:
                    feasibility = "Not feasible (exceeds GPU capacity)"
            else:
                if result['max_peak'] >= gpu_capacity:
                    feasibility = f"Insufficient memory (requires {result['max_peak']:.2f}GB, exceeds GPU capacity)"
                else:
                    feasibility = f"Partially successful (uses {usage_percentage:.1f}%)"

            print(f"  {result['chunk_size']}×{result['chunk_size']} chunks: {feasibility}")

    print(f"\nRecommended Configuration:")
    gpu_capacity = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

    successful_results = [r for r in training_results if r['success']]
    if successful_results:
        successful_results.sort(key=lambda x: x['max_peak'])
        best_result = successful_results[0]

        usage_percentage = best_result['max_peak'] / gpu_capacity * 100
        if usage_percentage < 40:
            recommendation = "Recommended for multi-tasking/multi-experiment parallel execution"
        elif usage_percentage < 70:
            recommendation = "Recommended for single experiment training"
        elif usage_percentage < 90:
            recommendation = "Recommended with memory monitoring"
        else:
            recommendation = "Not recommended, high risk"

        print(f"  Best configuration: {best_result['chunk_size']}×{best_result['chunk_size']} chunks")
        print(f"  Memory usage: {best_result['max_peak']:.2f}GB ({usage_percentage:.1f}%)")
        print(f"  Recommendation: {recommendation}")
    else:
        print("  Warning: All configuration tests failed!")


def main():
    print("Spatial Chunked Bi-GRU Complete Training Cycle Memory Test")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("GPU required for testing!")
        return

    test_full_training_cycle()


if __name__ == "__main__":
    main()