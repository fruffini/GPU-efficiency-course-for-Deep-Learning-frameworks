#!/usr/bin/env python3
"""
Exercise 11: Benchmark Different Caching Strategies (with MONAI UNet)
---------------------------------------------------------------------
This version replaces the simulated GPU load with a real MONAI UNet
training loop on 3D medical images, comparing:
    - Dataset (no cache)
    - CacheDataset (with different cache rates)
    - SmartCacheDataset
It benchmarks the actual end-to-end performance of each strategy.
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import psutil
import nibabel as nib
from monai.data import Dataset, CacheDataset, SmartCacheDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityRanged, RandRotate90d, ToTensord, Resized
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss


# ------------------------------------------------------------
# Define transforms
# ------------------------------------------------------------
def get_transforms():
    """Define MONAI transforms pipeline with FIXED output size"""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(96, 96, 96), mode="trilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(0, 2)),
        ToTensord(keys=["image"]),
    ])


# ------------------------------------------------------------
# Benchmark function
# ------------------------------------------------------------
def benchmark_dataset(dataset_class, data_dicts, transforms, name, num_epochs=2, **kwargs):
    print(f"\n{'=' * 70}")
    print(f"Benchmarking: {name}")
    print(f"{'=' * 70}")

    start_create = time.time()
    if dataset_class == Dataset:
        dataset = dataset_class(data=data_dicts, transform=transforms)
    else:
        dataset = dataset_class(data=data_dicts, transform=transforms, **kwargs)
    create_time = time.time() - start_create
    print(f"Dataset creation time: {create_time:.2f}s")

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)

    # --- Define GPU model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=1
    ).to(device)
    criterion = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # --- Training loop ---
    epoch_times = []
    for epoch in range(num_epochs):
        start_epoch = time.time()
        model.train()

        for batch_idx, batch in enumerate(dataloader):
            inputs = batch["image"].to(device)
            targets = torch.where(inputs > 0.5, 1.0, 0.0)  # fake segmentation target

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if batch_idx % 5 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}", end='\r')

        torch.cuda.synchronize()
        epoch_time = time.time() - start_epoch
        epoch_times.append(epoch_time)
        print(f"  Epoch {epoch+1} time: {epoch_time:.2f}s")

    avg_epoch_time = np.mean(epoch_times)
    memory_mb = psutil.Process().memory_info().rss / 1024**2
    gpu_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

    print(f"\nSummary for {name}:")
    print(f"  Avg epoch time: {avg_epoch_time:.2f}s")
    print(f"  Host memory: {memory_mb:.0f} MB | GPU memory: {gpu_mem:.0f} MB")

    del model, optimizer, criterion, scaler, dataloader
    torch.cuda.empty_cache()

    return {
        "name": name,
        "create_time": create_time,
        "epoch_times": epoch_times,
        "avg_epoch_time": avg_epoch_time,
        "memory_mb": memory_mb,
        "gpu_mb": gpu_mem
    }


# ------------------------------------------------------------
# Main function
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Exercise 11: Caching Strategies Benchmark (GPU UNet)")
    parser.add_argument("--cache-rates", "-c", default="1.0", help="Comma-separated cache rates for CacheDataset")
    parser.add_argument("--epochs", "-e", type=int, default=2)
    args = parser.parse_args()

    try:
        cache_rates = [float(x) for x in args.cache_rates.split(",") if x.strip() != ""]
        cache_rates = [r for r in cache_rates if 0.0 <= r <= 1.0]
    except ValueError:
        print("✗ ERROR: Invalid cache rates.")
        return

    data_dir = Path("chapter2_data_efficiency/data/nifti_samples")
    image_files = sorted(list(data_dir.glob("*.nii.gz")))
    if not image_files:
        print("❌ No NIfTI files found. Run: python chapter2_data_efficiency/generate_sample_data.py")
        return

    data_dicts = [{"image": str(f)} for f in image_files]
    transforms = get_transforms()

    print(f"✅ Found {len(image_files)} samples | caching rates = {cache_rates}")

    results = []

    # --- No cache ---
    results.append(benchmark_dataset(Dataset, data_dicts, transforms, "No Cache (Baseline)", num_epochs=args.epochs))

    # --- CacheDataset variants ---
    for rate in cache_rates:
        name = f"CacheDataset ({int(rate*100)}%)"
        res = benchmark_dataset(CacheDataset, data_dicts, transforms, name,
                                num_epochs=args.epochs, cache_rate=rate, num_workers=2)
        results.append(res)

    # --- SmartCache ---
    results.append(benchmark_dataset(SmartCacheDataset, data_dicts, transforms, "SmartCacheDataset",
                                     num_epochs=args.epochs, cache_num=len(data_dicts)//2,
                                     num_init_workers=2, num_replace_workers=1))

    # ------------------------------------------------------------
    # Plot Results
    # ------------------------------------------------------------
    Path("chapter2_data_efficiency/figures").mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (1) Epoch time per strategy
    ax = axes[0]
    for r in results:
        ax.plot(range(1, len(r["epoch_times"]) + 1), r["epoch_times"], "-o", label=r["name"])
    ax.set_title("Epoch Times per Strategy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Time (s)")
    ax.grid(True); ax.legend(fontsize=8)

    # (2) Average speedup
    baseline = results[0]["avg_epoch_time"]
    speedups = [baseline / r["avg_epoch_time"] for r in results]
    ax = axes[1]
    ax.bar([r["name"] for r in results], speedups, color="cornflowerblue", edgecolor="black")
    ax.axhline(y=1, color="r", linestyle="--", linewidth=1)
    ax.set_ylabel("Speedup × vs Baseline")
    ax.set_xticklabels([r["name"] for r in results], rotation=30, ha="right")
    ax.set_title("Average Speedup")

    # (3) Memory
    ax = axes[2]
    mem = [r["memory_mb"] for r in results]
    gpu = [r["gpu_mb"] for r in results]
    x = np.arange(len(results))
    ax.bar(x - 0.2, mem, width=0.4, label="Host RAM (MB)", color="teal")
    ax.bar(x + 0.2, gpu, width=0.4, label="GPU VRAM (MB)", color="orange")
    ax.set_xticks(x); ax.set_xticklabels([r["name"] for r in results], rotation=30, ha="right")
    ax.set_ylabel("Memory (MB)")
    ax.legend(fontsize=8); ax.set_title("Memory Usage")

    plt.tight_layout()
    plt.savefig("chapter2_data_efficiency/figures/caching_benchmark_gpu.png", dpi=150)
    plt.close()
    print("✅ Saved: chapter2_data_efficiency/figures/caching_benchmark_gpu.png")

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    best = max(results, key=lambda x: baseline / x["avg_epoch_time"])
    print("\n🏆 BEST STRATEGY:", best["name"])
    print(f"   • Speedup: {baseline / best['avg_epoch_time']:.2f}×")
    print(f"   • Host RAM: {best['memory_mb']:.0f} MB | GPU VRAM: {best['gpu_mb']:.0f} MB")
    print("\nRecommended usage:")
    print("  - CacheDataset(100%) if dataset < 50% RAM")
    print("  - SmartCacheDataset if dataset > RAM")
    print("  - NoCache only for debugging or massive datasets\n")


if __name__ == "__main__":
    main()

