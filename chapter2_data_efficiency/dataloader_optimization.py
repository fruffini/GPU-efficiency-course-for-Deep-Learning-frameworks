#!/usr/bin/env python3
"""
Exercise 13: DataLoader Optimization
------------------------------------
Benchmark the effect of different DataLoader settings:
  • num_workers
  • prefetch_factor
  • pin_memory
  • persistent_workers
on throughput, epoch time, and memory usage.
"""

import time
import torch
import psutil
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    ScaleIntensityRanged, RandRotate90d, ToTensord,
)


# ------------------------------------------------------------
# Define standard transforms
# ------------------------------------------------------------
def get_transforms():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200,
                             b_min=0.0, b_max=1.0, clip=True),
        RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(0, 2)),
        ToTensord(keys=["image"]),
    ])


# ------------------------------------------------------------
# Benchmark one configuration
# ------------------------------------------------------------
def benchmark_dataloader_config(
    dataset,
    batch_size=4,
    num_workers=0,
    prefetch_factor=2,
    pin_memory=False,
    persistent_workers=False,
    num_epochs=3
):
    config_name = f"workers={num_workers}, prefetch={prefetch_factor}, pin={pin_memory}"

    # Create DataLoader safely
    if num_workers == 0:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    print(f"\nTesting: {config_name}")
    print("-" * 60)

    epoch_times, batch_times_all = [], []
    process = psutil.Process()
    memory_start = process.memory_info().rss / 1024**2

    for epoch in range(num_epochs):
        start_epoch = time.time()
        batch_times = []
        for batch_idx, batch in enumerate(dataloader):
            start_batch = time.time()

            # Lightweight "GPU work"
            images = batch["image"]
            _ = images.mean()

            batch_times.append(time.time() - start_batch)
            if (batch_idx + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}", end='\r')

        epoch_time = time.time() - start_epoch
        epoch_times.append(epoch_time)
        batch_times_all.extend(batch_times)
        print(f"  Epoch {epoch+1}: {epoch_time:.2f}s")

    memory_end = process.memory_info().rss / 1024**2
    memory_delta = memory_end - memory_start

    avg_epoch_time = np.mean(epoch_times)
    avg_batch_time = np.mean(batch_times_all)
    throughput = batch_size * len(dataloader) / avg_epoch_time

    print(f"  Avg epoch: {avg_epoch_time:.2f}s | Throughput: {throughput:.2f} samples/s | ΔMem: {memory_delta:+.0f} MB")

    return {
        "config": config_name,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "pin_memory": pin_memory,
        "epoch_times": epoch_times,
        "avg_epoch_time": avg_epoch_time,
        "avg_batch_time": avg_batch_time,
        "throughput": throughput,
        "memory_delta_mb": memory_delta,
    }


# ------------------------------------------------------------
# Main benchmarking logic
# ------------------------------------------------------------
def main():
    print("=" * 70)
    print("Exercise 13: DataLoader Optimization")
    print("=" * 70)

    data_dir = Path("chapter2_data_efficiency/data/nifti_samples")
    image_files = sorted(list(data_dir.glob("*.nii.gz")))[:40]

    print(f"\nUsing {len(image_files)} samples")
    print(f"CPU cores: {psutil.cpu_count()} | RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")

    data_dicts = [{"image": str(f)} for f in image_files]
    transforms = get_transforms()

    print("\nCreating cached dataset (for fair DataLoader benchmarking)...")
    dataset = CacheDataset(data=data_dicts, transform=transforms, cache_rate=1.0, num_workers=4)
    print("✓ Dataset cached")

    configs_to_test = [
        {"num_workers": 0, "prefetch_factor": 2, "pin_memory": False},  # baseline
        {"num_workers": 2, "prefetch_factor": 2, "pin_memory": False},
        {"num_workers": 4, "prefetch_factor": 2, "pin_memory": False},
        {"num_workers": 8, "prefetch_factor": 2, "pin_memory": False},
        {"num_workers": 4, "prefetch_factor": 2, "pin_memory": True},
        {"num_workers": 4, "prefetch_factor": 4, "pin_memory": True},
    ]

    results = []
    print("\n" + "=" * 70)
    print("BENCHMARKING CONFIGURATIONS")
    print("=" * 70)

    for config in configs_to_test:
        r = benchmark_dataloader_config(
            dataset,
            batch_size=4,
            **config,
            persistent_workers=True,
            num_epochs=3,
        )
        results.append(r)

    # --------------------------------------------------------
    # Results summary
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    baseline = results[0]

    print(f"\n{'Configuration':<40} {'Epoch Time':<12} {'Speedup':<10} {'ΔMemory':<10}")
    print("-" * 70)
    for r in results:
        speed = baseline["avg_epoch_time"] / r["avg_epoch_time"]
        print(f"{r['config']:<40} {r['avg_epoch_time']:<12.2f} {speed:<10.2f}× {r['memory_delta_mb']:+8.0f} MB")

    best = max(results, key=lambda x: x["throughput"])

    print(f"\n🏆 BEST CONFIGURATION: {best['config']}")
    print(f"   Throughput: {best['throughput']:.2f} samples/s")
    print(f"   Speedup: {baseline['avg_epoch_time'] / best['avg_epoch_time']:.2f}×")
    print(f"   Memory Δ: {best['memory_delta_mb']:+.0f} MB")

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------
    print("\nGenerating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    Path("chapter2_data_efficiency/figures").mkdir(parents=True, exist_ok=True)

    # 1️⃣ Speedup vs num_workers
    ax = axes[0, 0]
    workers_results = [r for r in results if not r["pin_memory"]]
    workers = [r["num_workers"] for r in workers_results]
    speedups = [baseline["avg_epoch_time"] / r["avg_epoch_time"] for r in workers_results]
    ax.plot(workers, speedups, "-o", linewidth=2, markersize=8, color="steelblue")
    ax.set_xlabel("num_workers"); ax.set_ylabel("Speedup × vs Baseline")
    ax.set_title("Speedup vs num_workers", fontweight="bold")
    ax.grid(True, alpha=0.3); ax.axhline(1, color="r", ls="--", alpha=0.5)

    # 2️⃣ Throughput comparison
    ax = axes[0, 1]
    configs_short = [f"W={r['num_workers']}\nP={r['prefetch_factor']}\nPin={r['pin_memory']}" for r in results]
    throughputs = [r["throughput"] for r in results]
    colors = ["red" if i == 0 else ("green" if r["config"] == best["config"] else "steelblue")
              for i, r in enumerate(results)]
    bars = ax.bar(range(len(results)), throughputs, color=colors, edgecolor="black")
    for b, thr in zip(bars, throughputs):
        ax.text(b.get_x()+b.get_width()/2., thr, f"{thr:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Throughput (samples/s)"); ax.set_title("Throughput Comparison", fontweight="bold")
    ax.set_xticks(range(len(results))); ax.set_xticklabels(configs_short, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # 3️⃣ Epoch times
    ax = axes[1, 0]
    for r in [results[0], results[2], results[-1]]:
        ax.plot(range(1, len(r["epoch_times"])+1), r["epoch_times"], "-o", label=r["config"][:25])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Time (s)"); ax.legend(fontsize=8)
    ax.set_title("Epoch Time Evolution", fontweight="bold"); ax.grid(True, alpha=0.3)

    # 4️⃣ Memory deltas
    ax = axes[1, 1]
    mem_deltas = [r["memory_delta_mb"] for r in results]
    colors = ["green" if m < 300 else "orange" if m < 800 else "red" for m in mem_deltas]
    bars = ax.bar(range(len(results)), mem_deltas, color=colors, edgecolor="black")
    for b, m in zip(bars, mem_deltas):
        ax.text(b.get_x()+b.get_width()/2., m, f"{m:+.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("ΔMemory (MB)"); ax.set_title("Memory Overhead", fontweight="bold")
    ax.set_xticks(range(len(results))); ax.set_xticklabels(configs_short, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = "chapter2_data_efficiency/figures/dataloader_optimization.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"✓ Saved: {out}")

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    cpu_cores = psutil.cpu_count()
    ram_gb = psutil.virtual_memory().total / 1024**3

    print(f"""
System Summary:
  • CPU cores: {cpu_cores}
  • RAM: {ram_gb:.1f} GB

Training (recommended):
  DataLoader(
      batch_size=4,
      num_workers={min(4, cpu_cores // 2)},
      prefetch_factor=2,
      pin_memory=True,
      persistent_workers=True,
      shuffle=True
  )

Validation / Inference:
  DataLoader(
      batch_size=8,
      num_workers={min(2, cpu_cores // 2)},
      pin_memory=True,
      persistent_workers=False,
      shuffle=False
  )

Guidelines:
  1. Start with num_workers=4 → increase until GPU utilization >80%.
  2. Use pin_memory=True to speed up host→device transfer.
  3. prefetch_factor=2 usually sufficient (4 if GPU still idle).
  4. persistent_workers=True avoids DataLoader restarts each epoch.
  5. Monitor GPU util via nvidia-smi; aim for >80%.
""")
    print("=" * 70)


if __name__ == "__main__":
    main()
