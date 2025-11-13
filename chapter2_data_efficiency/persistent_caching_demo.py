#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Exercise 12 — Baseline vs Persistent vs Smart Cache Benchmark (GPU UNet)
========================================================================
Compares:
  •  Standard Dataset      → baseline, no caching
  •  PersistentDataset     → preprocessing cached on disk (persistent across runs)
  •  SmartCacheDataset     → dynamic RAM cache refreshed during training
"""

import time, shutil, psutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from monai.data import Dataset, PersistentDataset, SmartCacheDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    ScaleIntensityRanged, RandRotate90d, RandGaussianNoised,
    RandGaussianSmoothd, ToTensord, Resized
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss


# ------------------------------------------------------------
# Transforms — intentionally heavy to reveal caching effects
# ------------------------------------------------------------
def get_transforms():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1, 1, 1), mode="bilinear"),
        Resized(keys=["image"], spatial_size=(96, 96, 96), mode="trilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200,
                             b_min=0.0, b_max=1.0, clip=True),
        RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(0, 2)),
        RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
        RandGaussianSmoothd(keys=["image"], prob=0.3),
        ToTensord(keys=["image"]),
    ])


# ------------------------------------------------------------
# Generic training benchmark using MONAI UNet
# ------------------------------------------------------------
def benchmark(dataset, name, num_epochs=4):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=1
    ).to(device)
    criterion = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)
    epoch_times = []

    for e in range(num_epochs):
        t0 = time.time()
        for i, batch in enumerate(loader):
            x = batch["image"].to(device)
            y = (x > 0.5).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                out = model(x); loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            if i % 5 == 0:
                print(f"  Epoch {e+1} Batch {i+1}/{len(loader)} Loss={loss.item():.4f}", end="\r")
        torch.cuda.synchronize()
        t = time.time() - t0
        epoch_times.append(t)
        print(f"  Epoch {e+1} done in {t:.2f}s")

    avg = np.mean(epoch_times)
    cpu_mem = psutil.Process().memory_info().rss / 1024**2
    gpu_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    torch.cuda.empty_cache()
    return {"name": name, "epoch_times": epoch_times,
            "avg_epoch_time": avg, "cpu_mem": cpu_mem, "gpu_mem": gpu_mem}


# ------------------------------------------------------------
# PersistentDataset (first + resumed)
# ------------------------------------------------------------
def persistent_runs(data_dicts, cache_dir):
    tr = get_transforms()

    # First run (creates cache)
    t0 = time.time()
    ds_first = PersistentDataset(data_dicts, tr, cache_dir)
    build_time = time.time() - t0
    res_first = benchmark(ds_first, "PersistentDataset – First Run")

    # Resumed run (loads existing cache)
    t0 = time.time()
    ds_resume = PersistentDataset(data_dicts, tr, cache_dir)
    load_time = time.time() - t0
    res_resume = benchmark(ds_resume, "PersistentDataset – Resumed Run")

    cache_size = sum(f.stat().st_size for f in Path(cache_dir).rglob("*") if f.is_file())/1024**2
    for r, ct in zip((res_first, res_resume), (build_time, load_time)):
        r["cache_time"] = ct; r["cache_size"] = cache_size
    return res_first, res_resume


# ------------------------------------------------------------
# SmartCacheDataset benchmark
# ------------------------------------------------------------
def smartcache_run(data_dicts):
    tr = get_transforms()
    ds = SmartCacheDataset(data_dicts, tr, cache_num=len(data_dicts)//2,
                           num_init_workers=2, num_replace_workers=1)
    return benchmark(ds, "SmartCacheDataset (RAM cache)")


# ------------------------------------------------------------
# Baseline (no caching)
# ------------------------------------------------------------
def baseline_run(data_dicts):
    tr = get_transforms()
    ds = Dataset(data_dicts, tr)
    return benchmark(ds, "Baseline (No Cache)")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("="*70)
    print("Exercise 12 — Baseline vs Persistent vs Smart Cache")
    print("="*70)

    data_dir = Path("chapter2_data_efficiency/data/nifti_samples")
    imgs = sorted(list(data_dir.glob("*.nii.gz")))[:30]
    if not imgs:
        print("❌ No NIfTI files found. Run generate_sample_data.py first.")
        return
    data_dicts = [{"image": str(f)} for f in imgs]

    cache_dir = Path("chapter2_data_efficiency/cache_persistent")
    if cache_dir.exists(): shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    results = []
    results.append(baseline_run(data_dicts))
    p_first, p_resume = persistent_runs(data_dicts, cache_dir)
    results.extend([p_first, p_resume])
    results.append(smartcache_run(data_dicts))

    # ------------------ Plot results -------------------------
    Path("chapter2_data_efficiency/figures").mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Epoch times
    ax = axes[0]
    for r in results:
        ax.plot(range(1, len(r["epoch_times"])+1), r["epoch_times"], "-o", label=r["name"])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Time (s)")
    ax.set_title("Epoch Times per Strategy"); ax.legend(fontsize=7); ax.grid(True, alpha=.3)

    # 2. Speedup vs baseline
    base_t = results[0]["avg_epoch_time"]
    speedups = [base_t / r["avg_epoch_time"] for r in results]
    ax = axes[1]
    ax.bar([r["name"] for r in results], speedups, color="cornflowerblue", edgecolor="black")
    ax.axhline(1, color="r", ls="--"); ax.set_ylabel("Speedup × vs Baseline")
    ax.set_title("Average Speedup"); ax.set_xticklabels([r["name"] for r in results],
                                                       rotation=25, ha="right")

    # 3. Memory
    ax = axes[2]; x = np.arange(len(results))
    ax.bar(x-0.2, [r["cpu_mem"] for r in results], 0.4, label="Host RAM (MB)", color="teal")
    ax.bar(x+0.2, [r["gpu_mem"] for r in results], 0.4, label="GPU VRAM (MB)", color="orange")
    ax.set_xticks(x); ax.set_xticklabels([r["name"] for r in results], rotation=25, ha="right")
    ax.set_ylabel("Memory (MB)"); ax.legend(fontsize=8)
    ax.set_title("Memory Usage"); plt.tight_layout()

    out = "chapter2_data_efficiency/figures/cache_strategy_comparison.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"✅ Saved plot: {out}")

    # ------------------ Summary ------------------------------
    print("\n" + "="*70)
    print("SUMMARY & GUIDELINES")
    print("="*70)
    base = results[0]
    print(f"Baseline avg epoch: {base['avg_epoch_time']:.2f}s")
    for r in results[1:]:
        s = base["avg_epoch_time"]/r["avg_epoch_time"]
        print(f"{r['name']:<35} → {r['avg_epoch_time']:.2f}s  ({s:.2f}× faster)")
    if "cache_size" in results[1]:
        print(f"\nPersistent cache size ≈ {results[1]['cache_size']:.1f} MB")

    print("\n💡 When to use:")
    print("• Baseline – for debugging or when transforms are cheap.")
    print("• PersistentDataset – for heavy preprocessing reused across runs (long projects).")
    print("• SmartCacheDataset – for in-memory speed during a single run (augmentations).")
    print("\nRule of thumb:")
    print("  < RAM fits dataset  → SmartCacheDataset")
    print("  ≫ RAM but fixed pipeline → PersistentDataset")
    print("  Tiny/fast pipeline    → Baseline")
    print("="*70)


if __name__ == "__main__":
    main()


