
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from monai.networks.nets import BasicUNet


def count_conv3d_flops(in_channels, out_channels, kernel_size, input_shape):
    """Count FLOPs for a 3D convolution"""
    D, H, W = input_shape
    K_d, K_h, K_w = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3

    # FLOPs per output position: 2 ops (multiply + add) per kernel element
    flops_per_position = 2 * in_channels * K_d * K_h * K_w
    # Total output positions
    output_positions = D * H * W * out_channels

    return flops_per_position * output_positions


def estimate_model_flops(model, input_shape=(1, 1, 96, 96, 96)):
    """Estimate total FLOPs for BasicUNet"""
    batch, channels, D, H, W = input_shape

    # BasicUNet architecture: (16, 32, 64, 128, 256, 32)
    features = [16, 32, 64, 128, 256]

    total_flops = 0
    current_shape = (D, H, W)
    in_ch = channels

    # Encoder
    for feat in features:
        # Two conv3d per level
        total_flops += 2 * count_conv3d_flops(in_ch, feat, 3, current_shape)
        in_ch = feat
        # Downsample
        current_shape = tuple(s // 2 for s in current_shape)

    # Bottleneck
    total_flops += 2 * count_conv3d_flops(features[-1], features[-1], 3, current_shape)

    # Decoder
    for i in range(len(features) - 1, -1, -1):
        current_shape = tuple(s * 2 for s in current_shape)
        feat = features[i]
        # Upconv + two conv3d
        total_flops += count_conv3d_flops(in_ch, feat, 3, current_shape)
        total_flops += 2 * count_conv3d_flops(feat * 2, feat, 3, current_shape)
        in_ch = feat

    # Final conv
    total_flops += count_conv3d_flops(in_ch, 1, 1, current_shape)

    return total_flops * batch


def get_gpu_peak_flops():
    """Get theoretical peak FLOPS for current GPU"""
    device_name = torch.cuda.get_device_name(0).upper()

    # Peak TFLOPS for common GPUs (FP32)
    peak_flops_map = {
        "A100": 19.5e12,      # 19.5 TFLOPS FP32
        "V100": 15.7e12,      # 15.7 TFLOPS FP32
        "A6000": 38.7e12,     # 38.7 TFLOPS FP32
        "L40S": 91.6e12,      # 91.6 TFLOPS FP32 (with sparsity)
        "L40": 90.5e12,       # 90.5 TFLOPS FP32
        "H100": 60.0e12,      # 60 TFLOPS FP32
        "RTX 4090": 82.6e12,  # 82.6 TFLOPS FP32
        "RTX 3090": 35.6e12,  # 35.6 TFLOPS FP32
        "T4": 8.1e12,         # 8.1 TFLOPS FP32
    }

    for key, flops in peak_flops_map.items():
        if key in device_name:
            return flops

    # Default estimate
    return 20e12


def benchmark_flops(batch_size=2, vol_size=96, precision="fp32", iterations=20):
    """Benchmark model and compute achieved FLOPS"""

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    model = BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        features=(16, 32, 64, 128, 256, 32)
    ).to(device)

    if precision == "fp16":
        model = model.half()

    # Configure precision
    use_amp = precision in ["amp", "bf16"]
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    dtype = torch.float16 if precision == "fp16" else torch.float32
    x = torch.randn(batch_size, 1, vol_size, vol_size, vol_size,
                   device=device, dtype=dtype)


    # Benchmark
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(iterations):
        try:
            torch.cuda.synchronize()
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype):
                _ = model(x)

        except torch.OutOfMemoryError:
            print(f"OOM at batch size {batch_size}, volume size {vol_size}, precision {precision}")
            return {
                "precision": precision,
                "batch_size": batch_size,
                "vol_size": vol_size,
                "time_per_iter": float('inf'),
                "achieved_tflops": 0.0,
                "peak_tflops": 0.0,
                "efficiency_pct": 0.0,
                "memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "samples_per_sec": 0.0
            }
    elapsed = time.time() - start


    # Calculate FLOPS
    model_flops = estimate_model_flops(model, input_shape=(batch_size, 1, vol_size, vol_size, vol_size))
    total_flops = model_flops * iterations
    achieved_flops = total_flops / elapsed
    achieved_tflops = achieved_flops / 1e12

    # Peak FLOPS
    peak_flops = get_gpu_peak_flops()
    if precision in ["amp", "fp16", "bf16"]:
        peak_flops *= 2  # FP16 has 2× theoretical peak

    peak_tflops = peak_flops / 1e12
    efficiency = (achieved_flops / peak_flops) * 100

    # Memory bandwidth
    mem_used = torch.cuda.max_memory_allocated() / 1024**3

    return {
        "precision": precision,
        "batch_size": batch_size,
        "vol_size": vol_size,
        "time_per_iter": elapsed / iterations,
        "achieved_tflops": achieved_tflops,
        "peak_tflops": peak_tflops,
        "efficiency_pct": efficiency,
        "memory_gb": mem_used,
        "samples_per_sec": batch_size * iterations / elapsed
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FLOPS and Efficiency Analysis")
    parser.add_argument("--volume", type=int, default=96,
                        help="Input volume size (default: 96)")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    print("="*80)
    print("Exercise 9: FLOPS and Efficiency Analysis")
    print("="*80)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    peak_flops_fp32 = get_gpu_peak_flops() / 1e12
    peak_flops_fp16 = (get_gpu_peak_flops() * 2) / 1e12
    print(f"Peak FLOPS (FP32): {peak_flops_fp32:.1f} TFLOPS")
    print(f"Peak FLOPS (FP16): {peak_flops_fp16:.1f} TFLOPS")
    print("="*80 + "\n")


    # Test different configurations
    configs = [
        {"batch_size": 1, "vol_size": args.volume, "precision": "fp32"},
        {"batch_size": 2, "vol_size": args.volume, "precision": "fp32"},
        {"batch_size": 4, "vol_size": args.volume, "precision": "fp32"},
        {"batch_size": 8, "vol_size": args.volume, "precision": "fp32"},
        {"batch_size": 16, "vol_size": args.volume, "precision": "fp32"},
        {"batch_size": 1, "vol_size": args.volume, "precision": "amp"},
        {"batch_size": 2, "vol_size": args.volume, "precision": "amp"},
        {"batch_size": 4, "vol_size": args.volume, "precision": "amp"},
        {"batch_size": 8, "vol_size": args.volume, "precision": "amp"},
        {"batch_size": 16, "vol_size": args.volume, "precision": "amp"},
        {"batch_size": 1, "vol_size": args.volume, "precision": "bf16"},
        {"batch_size": 2, "vol_size": args.volume, "precision": "bf16"},
        {"batch_size": 4, "vol_size": args.volume, "precision": "bf16"},
        {"batch_size": 8, "vol_size": args.volume, "precision": "bf16"},
        {"batch_size": 16, "vol_size": args.volume, "precision": "bf16"},
    ]


    results = []

    print(f"{'Config':<20} {'TFLOPS':<10} {'Peak':<10} {'Efficiency':<12} {'Mem (GB)':<10}")
    print("-"*80)

    for config in configs:
        result = benchmark_flops(**config)
        results.append(result)

        config_str = f"{config['precision'].upper()}, B={config['batch_size']}, V={config['vol_size']}"
        print(f"{config_str:<20} {result['achieved_tflops']:<10.2f} "
              f"{result['peak_tflops']:<10.1f} {result['efficiency_pct']:<12.1f}% "
              f"{result['memory_gb']:<10.2f}")

    print("\n" + "="*80)
    print("Analysis Summary")
    print("="*80 + "\n")

    # Find best efficiency
    best_eff = max(results, key=lambda x: x['efficiency_pct'])
    print(f"🚀 Best GPU Efficiency: {best_eff['efficiency_pct']:.1f}%")
    print(f"   Config: {best_eff['precision'].upper()}, "
          f"Batch={best_eff['batch_size']}, Vol={best_eff['vol_size']}")
    print(f"   Achieved: {best_eff['achieved_tflops']:.2f} TFLOPS")
    print(f"   Peak: {best_eff['peak_tflops']:.1f} TFLOPS")
    print()

    # Find best throughput
    best_throughput = max(results, key=lambda x: x['samples_per_sec'])
    print(f"⚡ Best Throughput: {best_throughput['samples_per_sec']:.2f} samples/s")
    print(f"   Config: {best_throughput['precision'].upper()}, "
          f"Batch={best_throughput['batch_size']}, Vol={best_throughput['vol_size']}")
    print()

    # Memory efficiency
    print("💾 Memory Efficiency:")
    for result in results:
        samples_per_gb = result['samples_per_sec'] / result['memory_gb']
        print(f"   {result['precision'].upper()}, B={result['batch_size']}: "
              f"{samples_per_gb:.2f} samples/s/GB")
    print()

    # Create visualizations
    print("Generating plots...")

    # Plot 1: FLOPS by configuration
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # FLOPS comparison
    ax = axes[0, 0]
    configs_labels = [f"{r['precision'].upper()}\nB={r['batch_size']}" for r in results]
    achieved = [r['achieved_tflops'] for r in results]
    peak = [r['peak_tflops'] for r in results]

    x = np.arange(len(results))
    width = 0.35

    bars1 = ax.bar(x - width/2, achieved, width, label='Achieved',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, peak, width, label='Peak',
                   color='orange', alpha=0.8)

    ax.set_ylabel('TFLOPS', fontsize=11)
    ax.set_title('Achieved vs Peak FLOPS', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs_labels, fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Efficiency comparison
    ax = axes[0, 1]
    efficiency = [r['efficiency_pct'] for r in results]
    colors = ['green' if e > 50 else 'orange' if e > 30 else 'red' for e in efficiency]

    bars = ax.bar(x, efficiency, color=colors, alpha=0.7, edgecolor='black')
    for bar, eff in zip(bars, efficiency):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{eff:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('GPU Utilization (%)', fontsize=11)
    ax.set_title('GPU Efficiency', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs_labels, fontsize=9)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% target')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Throughput comparison
    ax = axes[1, 0]
    throughput = [r['samples_per_sec'] for r in results]
    bars = ax.bar(x, throughput, color='purple', alpha=0.7, edgecolor='black')

    for bar, thr in zip(bars, throughput):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{thr:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Samples per Second', fontsize=11)
    ax.set_title('Training Throughput', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs_labels, fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Memory usage
    ax = axes[1, 1]
    memory = [r['memory_gb'] for r in results]
    bars = ax.bar(x, memory, color='teal', alpha=0.7, edgecolor='black')

    for bar, mem in zip(bars, memory):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mem:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('GPU Memory (GB)', fontsize=11)
    ax.set_title('Memory Usage', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs_labels, fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'./chapter1_improve_the_model_performance/figures/flops_analysis_{args.volume}.png', dpi=150)
    plt.close()

    # Plot 2: Efficiency breakdown
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by precision
    fp32_results = [r for r in results if r['precision'] == 'fp32']
    amp_results = [r for r in results if r['precision'] == 'amp']
    bf16_results = [r for r in results if r['precision'] == 'bf16']

    fp32_batches = [r['batch_size'] for r in fp32_results]
    fp32_eff = [r['efficiency_pct'] for r in fp32_results]

    amp_batches = [r['batch_size'] for r in amp_results]
    amp_eff = [r['efficiency_pct'] for r in amp_results]

    bf16_batches = [r['batch_size'] for r in bf16_results]
    bf16_eff = [r['efficiency_pct'] for r in bf16_results]

    ax.plot(fp32_batches, fp32_eff, '-o', linewidth=2, markersize=10,
            label='FP32', color='steelblue')
    ax.plot(amp_batches, amp_eff, '-s', linewidth=2, markersize=10,
            label='AMP', color='orange')
    ax.plot(bf16_batches, bf16_eff, '-^', linewidth=2, markersize=10,
            label='BF16', color='green')

    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('GPU Efficiency (%)', fontsize=12)
    ax.set_title('GPU Efficiency vs Batch Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% target')

    plt.tight_layout()
    plt.savefig(f'./chapter1_improve_the_model_performance/figures/efficiency_vs_batch_{args.volume}.png', dpi=150)
    plt.close()

    print("✓ Saved: figures/flops_analysis.png")
    print("✓ Saved: figures/efficiency_vs_batch.png")

    print("\n" + "="*80)
    print("Key Insights")
    print("="*80)

    print("""
1. GPU EFFICIENCY:
   • >70% = Excellent (compute-bound, GPU well utilized)
   • 40-70% = Good (some optimization possible)
   • <40% = Poor (CPU bottleneck or small batches)

2. MEMORY vs COMPUTE:
   • Small batches: Memory underutilized, low FLOPS
   • Large batches: Better FLOPS, but may OOM
   • Sweet spot: Batch size that achieves >50% efficiency

3. PRECISION IMPACT:
   • FP32: Lower efficiency, higher memory
   • AMP/FP16: Higher efficiency, lower memory
   • Always use AMP for 3D medical imaging!

4. OPTIMIZATION STRATEGY:
   • Start with small batch, measure efficiency
   • Increase batch until efficiency peaks or OOM
   • Use AMP to double effective compute capacity
   • Monitor: aim for >50% GPU utilization
    """)

    print("="*80)


if __name__ == "__main__":
    main()