#!/usr/bin/env python3
"""
Exercise 7: Precision Comparison Benchmark
"""

import torch
import time
import gc
import matplotlib.pyplot as plt
from monai.networks.nets import BasicUNet
from monai.losses import DiceLoss


torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

capability = torch.cuda.get_device_capability(0)
supports_bf16 = capability >= (8, 0)
supports_fp16 = capability >= (7, 0)

print("="*70)
print("Exercise 7: Precision Benchmark")
print("="*70)
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Compute: {capability[0]}.{capability[1]}")
print(f"BF16: {supports_bf16}, FP16: {supports_fp16}")
print("="*70 + "\n")

precisions = ["fp32", "amp"]
if supports_bf16:
    precisions.append("bf16")
if supports_fp16:
    precisions.append("fp16")

batch_size = 2
vol_size = 96
steps = 10

results = []

print(f"{'Precision':<10} {'Memory':<12} {'Time/iter':<12} {'Samples/s':<12} {'Status':<10}")
print("-"*70)

for prec in precisions:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        features=(32, 32, 64, 128, 256, 32)
    ).to(device)

    criterion = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Configure precision
    if prec == "fp32":
        use_amp = False
        use_scaler = False
        dtype = torch.float32
        autocast_dtype = None
    elif prec == "amp":
        use_amp = True
        use_scaler = True
        dtype = torch.float32
        autocast_dtype = torch.float16
    elif prec == "bf16":
        use_amp = True
        use_scaler = False
        dtype = torch.float32
        autocast_dtype = torch.bfloat16
    elif prec == "fp16":
        use_amp = False
        use_scaler = False
        dtype = torch.float16
        autocast_dtype = None
        model = model.half()

    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    x = torch.randn(batch_size, 1, vol_size, vol_size, vol_size, device=device, dtype=dtype)
    y = torch.randint(0, 2, (batch_size, 1, vol_size, vol_size, vol_size),
                     device=device, dtype=dtype)

    # Warmup
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype):
        _ = model(x)

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype):
            out = model(x)
            loss = criterion(out, y)

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()

    elapsed = (time.time() - start) / steps
    mem = torch.cuda.max_memory_allocated() / 1024**2
    samples_s = batch_size / elapsed

    results.append((prec, mem, elapsed, samples_s))
    print(f"{prec.upper():<10} {mem:<12.0f} {elapsed:<12.3f} {samples_s:<12.2f} {'✓':<10}")

    del model, x, y, out, loss
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

print("\n" + "="*70)

# Plot results
prec_labels, mem_vals, time_vals, samp_vals = zip(*results)
colors = ["steelblue", "orange", "green", "purple"][:len(prec_labels)]

# Memory plot
plt.figure(figsize=(10, 6))
bars = plt.bar(prec_labels, mem_vals, color=colors, alpha=0.8, edgecolor='black')
for bar, mem in zip(bars, mem_vals):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{mem:.0f}', ha='center', va='bottom', fontweight='bold')

plt.title("GPU Memory by Precision", fontsize=14, fontweight='bold')
plt.ylabel("Memory (MB)", fontsize=12)
plt.xlabel("Precision", fontsize=12)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("./chapter1_improve_the_model_performance/figures/precision_memory.png", dpi=150)
plt.close()

# Throughput plot
plt.figure(figsize=(10, 6))
bars = plt.bar(prec_labels, samp_vals, color=colors, alpha=0.8, edgecolor='black')
for bar, samp in zip(bars, samp_vals):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{samp:.1f}', ha='center', va='bottom', fontweight='bold')

plt.title("Throughput by Precision", fontsize=14, fontweight='bold')
plt.ylabel("Samples/s", fontsize=12)
plt.xlabel("Precision", fontsize=12)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("./chapter1_improve_the_model_performance/figures/precision_throughput.png", dpi=150)
plt.close()

print("✓ Plots saved")

best = max(results, key=lambda x: x[3])
print(f"\n🚀 Best: {best[0].upper()} ({best[1]:.0f} MB, {best[3]:.2f} samp/s)")

# Calculate speedups
fp32_time = next(t for p, _, t, _ in results if p == "fp32")
print("\nSpeedup vs FP32:")
for prec, _, t, _ in results:
    speedup = fp32_time / t
    print(f"  {prec.upper()}: {speedup:.2f}×")
