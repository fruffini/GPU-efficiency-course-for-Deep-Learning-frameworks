#!/usr/bin/env python3
"""
Exercise 14: Prefetch + Persistent Interaction
------------------------------------------------
Benchmark combined effects of persistent_workers and prefetch_factor.
"""
#!/usr/bin/env python3
"""
Exercise 14b: DataLoader Optimization with Persistent Workers & Prefetch
------------------------------------------------------------------------
Evaluates interaction of num_workers, pin_memory, prefetch_factor,
and persistent_workers on real GPU training throughput using MONAI BasicUNet.
"""

import time
import torch
import psutil
import itertools
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import BasicUNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    ScaleIntensityRanged, RandRotate90d, ToTensord,
)

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Transform pipeline
# ============================================================
def get_transforms():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200,
                             b_min=0.0, b_max=1.0, clip=True),
        RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 2)),
        ToTensord(keys=["image"]),
    ])


# ============================================================
# Benchmark function
# ============================================================
def benchmark_loader(model, dataset, num_workers, pin_memory,
                     persistent_workers, prefetch_factor, batch_size=2, epochs=2):
    """Benchmark a single DataLoader configuration."""
    name = f"{num_workers}w pin={pin_memory} persist={persistent_workers} prefetch={prefetch_factor}"
    process = psutil.Process()

    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor

    loader = DataLoader(**kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    mem0 = process.memory_info().rss / 1024**2
    epoch_times = []

    for e in range(epochs):
        start = time.time()
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = torch.zeros_like(x)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
        epoch_times.append(time.time() - start)

    mem1 = process.memory_info().rss / 1024**2
    avg_epoch = np.mean(epoch_times)
    throughput = batch_size * len(loader) / avg_epoch

    print(f"{name:55s} | {avg_epoch:6.2f}s | {throughput:7.1f} samples/s | ΔMem {mem1 - mem0:+.0f} MB")
    return dict(
        name=name,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        avg_epoch=avg_epoch,
        throughput=throughput,
        mem_delta=mem1 - mem0,
    )


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("Exercise 14b: Persistent Workers + Prefetch Benchmark")
    print("=" * 70)

    data_dir = Path("chapter2_data_efficiency/data/nifti_samples")
    image_files = sorted(list(data_dir.glob("*.nii.gz")))
    if len(image_files) == 0:
        print("❌ No NIfTI files found! Run generate_sample_nifti.py first.")
        return

    data_dicts = [{"image": str(f)} for f in image_files]
    dataset = CacheDataset(data=data_dicts, transform=get_transforms(),
                           cache_rate=1.0, num_workers=4)
    print(f"✓ Using cached dataset with {len(dataset)} samples")

    model = BasicUNet(spatial_dims=3, in_channels=1, out_channels=1).to(device)

    # Define configurations grid
    num_workers_list = [1, 2, 4]
    pin_memory_list = [False, True]
    persist_list = [False, True]
    prefetch_list = [1, 2, 4]

    results = []
    for nw, pin, pw, pf in itertools.product(num_workers_list, pin_memory_list, persist_list, prefetch_list):
        if nw == 0 and pf != 2:
            continue  # prefetch not used for workers=0
        results.append(benchmark_loader(model, dataset, nw, pin, pw, pf))

    Path("chapter2_data_efficiency/figures").mkdir(parents=True, exist_ok=True)

    # --- Organize data ---
    baseline_thr = [r for r in results if r["num_workers"] == 0][0]["throughput"]
    results_sorted = sorted(results, key=lambda r: (r["num_workers"], r["pin_memory"], r["persistent_workers"], r["prefetch_factor"]))

    # ============================================================
    # Plot 1: Throughput grouped by Prefetch
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {1: "#9fc5e8", 2: "#3d85c6", 4: "#0b5394"}
    for pf in prefetch_list:
        vals = [r["throughput"] for r in results if r["prefetch_factor"] == pf and r["num_workers"] > 0]
        nworkers = [r["num_workers"] for r in results if r["prefetch_factor"] == pf and r["num_workers"] > 0]
        ax.plot(nworkers, vals, '-o', label=f"Prefetch={pf}", color=colors[pf], lw=2)
    ax.set_xlabel("Number of Workers")
    ax.set_ylabel("Throughput (samples/s)")
    ax.set_title("Throughput vs num_workers for Different Prefetch Factors")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("chapter2_data_efficiency/figures/prefetch_throughput.png", dpi=150)
    plt.close()

    # ============================================================
    # Plot 2: Speedup vs Baseline
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [r["name"] for r in results_sorted]
    speedups = [r["throughput"] / baseline_thr for r in results_sorted]
    ax.bar(range(len(results_sorted)), speedups, color="orange", alpha=0.8)
    ax.set_xticks(range(len(results_sorted)))
    ax.set_xticklabels(names, rotation=80, fontsize=7, ha="right")
    ax.set_ylabel("Speedup vs workers=0")
    ax.set_title("Relative Speedup from Prefetch, Pin, and Persistence")
    plt.tight_layout()
    plt.savefig("chapter2_data_efficiency/figures/prefetch_speedup.png", dpi=150)
    plt.close()

    # ============================================================
    # Plot 3: Memory Usage
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#6fa8dc" if "persist=False" in r["name"] else "#0b5394" for r in results_sorted]
    ax.barh(range(len(results_sorted)), [r["mem_delta"] for r in results_sorted],
            color=colors, alpha=0.8, edgecolor="black")
    ax.set_yticks(range(len(results_sorted)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Δ Memory (MB)")
    ax.set_title("Memory Cost of Persistent Workers + Prefetch")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("chapter2_data_efficiency/figures/prefetch_memory.png", dpi=150)
    plt.close()

    # ============================================================
    # Summary & Guide
    # ============================================================
    print("\n✅ Saved figures:")
    print("  • prefetch_throughput.png")
    print("  • prefetch_speedup.png")
    print("  • prefetch_memory.png")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
🔹 Prefetch=1 → minimal buffering (safe but CPU-idle often)
🔹 Prefetch=2 → balanced (default in PyTorch)
🔹 Prefetch=4 → maximizes overlap, slightly higher RAM

🔹 persistent_workers=True → avoids worker respawn overhead each epoch
🔹 pin_memory=True → faster CPU→GPU transfers
🔹 num_workers=4 typically best trade-off for GPU saturation

💡 Key Takeaway:
   Increasing prefetch + persistent_workers helps when:
     • Each batch takes ≥100ms on GPU
     • Dataset fits in memory
   Too many workers or prefetch=8 can **reduce** throughput (CPU thrash)
""")
    print("=" * 70)

    # ============================================================
    # User Guide Echo
    # ============================================================
    print(r"""
⏱️  ESTIMATED TIME: 8–12 minutes

This benchmark will:
  1. Create cached dataset (~2 min)
  2. Test ~20 DataLoader configurations (~8–10 min)
  3. Generate performance plots automatically

WHAT TO WATCH:
  • num_workers=0 is slowest (baseline)
  • More workers + higher prefetch = better overlap
  • pin_memory=True gives ~10% faster transfers
  • persistent_workers=True avoids respawn delays

MONITOR IN ANOTHER TERMINAL:
  watch -n 1 "ps aux | grep python | wc -l"
  → You'll see worker processes persist!

EXPECTED SPEEDUP:
  • num_workers=4 + prefetch=4 → 2–5× faster than baseline
  • + pin_memory=True → +10–15% speedup
  • + persistent_workers=True → +10–20% smoother epochs
""")
    print("=" * 70)


if __name__ == "__main__":
    main()

