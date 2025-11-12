#!/usr/bin/env python3
"""
Exercise 8: Precision + Gradient Accumulation
"""

import torch
import time
import gc
import itertools
import matplotlib.pyplot as plt
import numpy as np
from monai.networks.nets import BasicUNet
from monai.losses import DiceLoss

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("Exercise 8: Precision + Gradient Accumulation")
print("=" * 70)
print(f"Device: {torch.cuda.get_device_name(0)}")
print("=" * 70 + "\n")

precisions = ["fp32", "amp", "bf16"]
accum_steps_list = [1, 2, 4, 8, 16]
batch_size = 2
vol_size = 96
total_samples = 12

results = []

print(f"{'Prec':<6} {'Accum':<7} {'Memory':<12} {'Time':<10} {'Samples/s':<12} {'Status':<10}")
print("-" * 70)

for prec, accum in itertools.product(precisions, accum_steps_list):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = BasicUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            features=(16, 32, 64, 128, 256, 32)
    ).to(device)

    criterion = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Configure precision
    if prec == "fp32":
        autocast_enabled = False
        autocast_dtype = torch.float32
        use_scaler = False
    elif prec == "amp":
        autocast_enabled = True
        autocast_dtype = torch.float16
        use_scaler = True
    elif prec == "bf16":
        autocast_enabled = True
        autocast_dtype = torch.bfloat16
        use_scaler = False

    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    x = torch.randn(batch_size, 1, vol_size, vol_size, vol_size, device=device)
    y = torch.randint(0, 2, (batch_size, 1, vol_size, vol_size, vol_size),
                      device=device, dtype=torch.float32)

    steps = total_samples // batch_size

    torch.cuda.synchronize()
    start = time.time()

    for i in range(steps):
        optimizer.zero_grad(set_to_none=True)

        for _ in range(accum):
            with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=autocast_dtype):
                out = model(x)
                loss = criterion(out, y) / accum

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        torch.cuda.synchronize()

    elapsed = time.time() - start
    mem = torch.cuda.max_memory_allocated() / 1024 ** 2
    samples_s = (batch_size * steps * accum) / elapsed

    results.append((prec, accum, mem, elapsed, samples_s))
    print(f"{prec.upper():<6} {accum:<7} {mem:<12.0f} {elapsed:<10.2f} {samples_s:<12.2f} {'✓':<10}")

    del model, x, y, out, loss
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

print("\n" + "=" * 70)

# Plot results
prec_vals, accum_vals, mem_vals, time_vals, samp_vals = zip(*results)

# Memory vs accumulation
plt.figure(figsize=(10, 6))
for prec in sorted(set(prec_vals)):
    xs = [a for p, a, _, _, _ in results if p == prec]
    ys = [m for p, a, m, _, _ in results if p == prec]
    plt.plot(xs, ys, '-o', linewidth=2, markersize=8, label=f'{prec.upper()}')

plt.title("Memory vs Gradient Accumulation", fontsize=14, fontweight='bold')
plt.xlabel("Accumulation Steps", fontsize=12)
plt.ylabel("Memory (MB)", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./chapter1_improve_the_model_performance/figures/memory_accumulation.png", dpi=150)
plt.close()

# Throughput vs accumulation
plt.figure(figsize=(10, 6))
for prec in sorted(set(prec_vals)):
    xs = [a for p, a, _, _, _ in results if p == prec]
    ys = [s for p, a, _, _, s in results if p == prec]
    plt.plot(xs, ys, '-o', linewidth=2, markersize=8, label=f'{prec.upper()}')

plt.title("Throughput vs Gradient Accumulation", fontsize=14, fontweight='bold')
plt.xlabel("Accumulation Steps", fontsize=12)
plt.ylabel("Samples/s", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./chapter1_improve_the_model_performance/figures/throughput_accumulation.png", dpi=150)
plt.close()

print("✓ Plots saved")

best = max(results, key=lambda x: x[4])
print(f"\n🚀 Best: {best[0].upper()}, accum={best[1]}")
print(f"   Memory: {best[2]:.0f} MB, Throughput: {best[4]:.2f} samp/s")

