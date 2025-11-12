#!/usr/bin/env python3
"""
Exercise 6: GPU Throughput Scaling Benchmark
"""

import torch
import time
import gc
import itertools
import matplotlib.pyplot as plt
from monai.networks.nets import BasicUNet
from monai.losses import DiceLoss


torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_sizes = [1, 2, 4, 8]
vol_sizes = [64, 96, 128, 168, 192]
total_samples_per_epoch = 12
results = []

print("="*70)
print("Exercise 6: GPU Scaling Benchmark")
print("="*70)
print(f"Device: {torch.cuda.get_device_name(0)}")
print("="*70 + "\n")

print(f"{'Batch':<6} {'Volume':<8} {'Memory':<12} {'Time':<10} {'Samples/s':<12} {'Status':<10}")
print("-"*70)
for bs, v in itertools.product(batch_sizes, vol_sizes):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        model = BasicUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            features=(16, 32, 64, 128, 256, 32)
        ).to(device)

        criterion = DiceLoss(sigmoid=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        steps_per_epoch = total_samples_per_epoch // bs
        x = torch.randn(bs, 1, v, v, v, device=device)
        y = torch.randint(0, 2, (bs, 1, v, v, v), device=device, dtype=torch.float32)

        # Warmup
        with torch.no_grad():
            _ = model(x)

        torch.cuda.synchronize()
        start = time.time()

        for _ in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                out = model(x)
                loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

        elapsed = time.time() - start
        mem = torch.cuda.max_memory_allocated() / 1024**2
        samples_per_sec = (bs * steps_per_epoch) / elapsed

        results.append((bs, v, mem, elapsed, samples_per_sec))
        print(f"{bs:<6} {v:<8} {mem:<12.0f} {elapsed:<10.2f} {samples_per_sec:<12.2f} {'✓':<10}")

    except torch.cuda.OutOfMemoryError:
        # Handle OOM gracefully
        results.append((bs, v, None, None, None))
        print(f"{bs:<6} {v:<8} {'OOM':<12} {'-':<10} {'-':<12} {'✗ OOM':<10}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        continue  # Skip to next configuration

    finally:
        # Always release GPU memory
        del model, x, y
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

print("\n" + "="*70)

# Plot results
valid = [(b, v, m, e, s) for b, v, m, e, s in results if m is not None]

# Memory plot
plt.figure(figsize=(10, 6))
for bs in sorted(set(b for b, _, _, _, _ in valid)):
    xs = [v for b, v, _, _, _ in valid if b == bs]
    ys = [m for b, v, m, _, _ in valid if b == bs]
    plt.plot(xs, ys, '-o', linewidth=2, markersize=8, label=f'Batch {bs}')

plt.title("GPU Memory vs Volume Size", fontsize=14, fontweight='bold')
plt.xlabel("Volume Size (voxels per side)", fontsize=12)
plt.ylabel("Max GPU Memory (MB)", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./chapter1_improve_the_model_performance/figures/memory_scaling.png", dpi=150)
plt.close()

# Throughput plot
plt.figure(figsize=(10, 6))
for bs in sorted(set(b for b, _, _, _, _ in valid)):
    xs = [v for b, v, _, _, _ in valid if b == bs]
    ys = [s for b, v, _, _, s in valid if b == bs]
    plt.plot(xs, ys, '-o', linewidth=2, markersize=8, label=f'Batch {bs}')

plt.title("Throughput vs Volume Size", fontsize=14, fontweight='bold')
plt.xlabel("Volume Size (voxels per side)", fontsize=12)
plt.ylabel("Samples per Second", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./chapter1_improve_the_model_performance/figures/throughput_scaling.png", dpi=150)
plt.close()

print("✓ Plots saved to ./chapter1_improve_the_model_performance/figures/")

best = max(valid, key=lambda x: x[4])
print(f"\n🚀 Optimal: Batch={best[0]}, Vol={best[1]}³")
print(f"   Memory: {best[2]:.0f} MB, Throughput: {best[4]:.2f} samples/s")
