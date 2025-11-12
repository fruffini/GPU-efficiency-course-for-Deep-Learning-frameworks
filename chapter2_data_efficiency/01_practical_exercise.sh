#!/bin/bash

# ================================================================
# CHAPTER 2 — DATA LOADING AND CACHING OPTIMIZATION
# Interactive Tutorial with Step-by-Step Guidance
# ================================================================

set -e  # Exit on error

# Color codes for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo ""
    echo -e "${CYAN}================================================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}================================================================${NC}"
    echo ""
}

print_section() {
    echo ""
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}$1${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_command() {
    echo -e "${GREEN}$${NC} ${CYAN}$1${NC}"
}

pause() {
    echo ""
    read -p "Press ENTER to continue..."
    echo ""
}

pause_optional() {
    echo ""
    read -p "Press ENTER to continue (or Ctrl+C to exit)..."
    echo ""
}

# ================================================================
# WELCOME SCREEN
# ================================================================

clear
print_header "CHAPTER 2: DATA LOADING AND CACHING OPTIMIZATION"

cat << 'EOF'
    ____        _          _                    _ _
   |  _ \  __ _| |_ __ _  | |    ___   __ _  __| (_)_ __   __ _
   | | | |/ _` | __/ _` | | |   / _ \ / _` |/ _` | | '_ \ / _` |
   | |_| | (_| | || (_| | | |__| (_) | (_| | (_| | | | | | (_| |
   |____/ \__,_|\__\__,_| |_____\___/ \__,_|\__,_|_|_| |_|\__, |
                                                            |___/
EOF

echo ""
echo -e "${CYAN}Welcome to Chapter 2: Data Loading and Caching Optimization${NC}"
echo ""
echo "In this chapter, you will learn:"
echo "  • Why data loading is the #1 bottleneck in medical imaging"
echo "  • How to use MONAI's caching strategies"
echo "  • Persistent caching for multi-session training"
echo "  • DataLoader optimization (num_workers, prefetching)"
echo "  • Memory-mapped arrays for large datasets"
echo "  • Dataset state management and checkpointing"
echo ""
echo "Prerequisites:"
echo "  ✓ Completed Chapter 1 (Model optimization)"
echo "  ✓ MONAI installed"
echo "  ✓ At least 32GB RAM (for caching examples)"
echo "  ✓ 10GB free disk space (for persistent cache)"
echo ""

pause

# ================================================================
# SETUP
# ================================================================

print_header "STEP 0: Environment Setup"

print_section "Creating Directory Structure"

mkdir -p chapter2_data_efficiency/figures
mkdir -p chapter2_data_efficiency/data/nifti_samples
mkdir -p chapter2_data_efficiency/cache_persistent
mkdir -p chapter2_data_efficiency/logs

print_success "Directories created"
print_info "  • chapter2_data_efficiency/data/ - sample medical images"
print_info "  • chapter2_data_efficiency/cache_persistent/ - persistent cache storage"
print_info "  • chapter2_data_efficiency/figures/ - benchmark plots"
print_info "  • chapter2_data_efficiency/logs/ - profiling logs"

echo ""

print_section "Checking Environment"

python3 << 'PYEOF'
import sys

packages = {
    'torch': 'PyTorch',
    'monai': 'MONAI',
    'nibabel': 'NiBabel',
    'matplotlib': 'Matplotlib',
    'numpy': 'NumPy'
}

missing = []
for pkg, name in packages.items():
    try:
        __import__(pkg)
        print(f"✓ {name}")
    except ImportError:
        print(f"✗ {name} - MISSING")
        missing.append(pkg)

if missing:
    print(f"\nInstall missing packages:")
    print(f"  pip install {' '.join(missing)}")
    sys.exit(1)

print("\n✓ All packages available")
PYEOF

pause

# ================================================================
# CONCEPT INTRODUCTION
# ================================================================

print_header "THE DATA LOADING PROBLEM"

cat << 'EOF'
📚 WHY DATA LOADING IS CRITICAL IN MEDICAL IMAGING

TYPICAL SCENARIO:
  • 3D CT scan: 512×512×300 voxels = ~300 MB per volume
  • Training dataset: 1000 scans = 300 GB total
  • GPU can process a batch in 0.5 seconds
  • Loading from disk takes 2-3 seconds per volume!

RESULT: GPU sits idle 80% of the time waiting for data! 💤

┌─────────────────────────────────────────────────────────┐
│ Without Optimization:                                    │
│ GPU: ████░░░░░░░░░░░░░░░░░░░░░░░  20% utilized         │
│ CPU: ░░░░████████████████████████  Data loading busy   │
│ Disk: ████████████████████████████ Reading from SSD    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ With Optimization (Caching + Prefetching):              │
│ GPU: ████████████████████████████  95% utilized ✓      │
│ CPU: ████░░░░░░░░░░░░░░░░░░░░░░░  Minimal overhead    │
│ Cache: ████████████████████████  Serving from RAM      │
└─────────────────────────────────────────────────────────┘

OPTIMIZATION STRATEGIES (This Chapter):

1. CACHING (Exercise 11):
   • Load data once, keep in RAM
   • 10-100× faster than disk access
   • Trade-off: Memory vs Speed

2. PERSISTENT CACHING (Exercise 12):
   • Pre-process once, save to disk
   • Resume training without reloading
   • Perfect for multi-day training

3. SMART DATALOADER (Exercise 13):
   • Multi-worker parallel loading
   • Prefetch next batch while GPU trains
   • Pin memory for faster GPU transfer


EOF

pause

# ================================================================
# EXERCISE 11: CACHING STRATEGIES
# ================================================================

print_header "EXERCISE 11: MONAI Caching Strategies"

cat << 'EOF'
📚 CONCEPT: Caching in Medical Imaging

PROBLEM:
  Medical images are large (100-500 MB per scan)
  Loading from disk is SLOW (2-5 seconds per volume)
  GPU training is FAST (0.1-0.5 seconds per batch)

  → GPU starves waiting for data!

SOLUTION: Keep processed data in RAM (caching)

MONAI CACHING LEVELS:

1. NO CACHE (Baseline):
   • Read from disk every epoch
   • Slowest, but zero memory overhead
   • Use: When memory is extremely limited

2. CACHE DATASET:
   • Load all data once, keep in RAM
   • ~10× faster than no cache
   • Memory needed: dataset_size × volume_size
   • Use: When dataset fits in RAM

3. SMART CACHE DATASET:
   • Cache most frequently accessed samples
   • Adaptive replacement policy
   • Memory needed: configurable (e.g., 50% of dataset)
   • Use: When dataset is larger than RAM

EXAMPLE MEMORY CALCULATION:
  Dataset: 500 CT scans
  Volume size: 128³ × 4 bytes (float32) = 8.4 MB
  Total cached: 500 × 8.4 MB = 4.2 GB

  → Feasible on most workstations!
EOF

pause

print_section "Creating Exercise 11 Scripts"

print_info "Generating demo data generator..."

cat > chapter2_data_efficiency/generate_sample_data.py << 'PYEOF'
#!/usr/bin/env python3
"""
Generate sample NIfTI medical images for benchmarking
"""

import numpy as np
import nibabel as nib
import os
from pathlib import Path
import argparse


def generate_sample_nifti(output_dir, num_samples=20, fixed_shape=(128, 128, 128)):
    """Generate synthetic medical images with FIXED shape"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_samples} sample NIfTI files...")
    print(f"Fixed shape: {fixed_shape}")

    for i in range(num_samples):
        # Create realistic-looking medical image with FIXED shape
        shape = fixed_shape  # Always use same shape

        # Background
        data = np.random.randn(*shape).astype(np.float32) * 50 + 100

        # Add some "organs" (spheres and ellipsoids)
        center = np.array(shape) // 2
        for _ in range(5):
            organ_center = center + np.random.randint(-30, 30, 3)
            radius = np.random.randint(10, 25)

            # Create sphere
            coords = np.ogrid[:shape[0], :shape[1], :shape[2]]
            distance = np.sqrt(
                (coords[0] - organ_center[0])**2 +
                (coords[1] - organ_center[1])**2 +
                (coords[2] - organ_center[2])**2
            )
            mask = distance < radius
            data[mask] = np.random.randint(150, 250)

        # Add some noise
        data += np.random.randn(*shape) * 10

        # Clip to realistic HU range
        data = np.clip(data, -1000, 1000)

        # Create NIfTI image with consistent affine
        affine = np.diag([1.5, 1.5, 2.0, 1.0])  # Consistent spacing
        img = nib.Nifti1Image(data, affine=affine)

        # Save
        filename = output_dir / f"sample_{i:03d}.nii.gz"
        nib.save(img, filename)

        if (i + 1) % 5 == 0:
            print(f"  Generated {i + 1}/{num_samples}...")

    print(f"✓ Generated {num_samples} samples in {output_dir}")
    print(f"  Total size: {sum(f.stat().st_size for f in output_dir.glob('*.nii.gz')) / 1024**2:.1f} MB")

    # Verify all files have same shape
    print("\nVerifying shapes...")
    shapes = []
    for f in sorted(output_dir.glob("*.nii.gz")):
        img = nib.load(f)
        shapes.append(img.shape)

    unique_shapes = set(shapes)
    if len(unique_shapes) == 1:
        print(f"✓ All files have consistent shape: {list(unique_shapes)[0]}")
    else:
        print(f"⚠ Warning: Found {len(unique_shapes)} different shapes: {unique_shapes}")


def main():
    parser = argparse.ArgumentParser(description="Generate sample NIfTI files for benchmarking")
    parser.add_argument("--output-dir", "-o", default="chapter2_data_efficiency/data/nifti_samples",
                        help="Directory to write sample NIfTI files")
    parser.add_argument("--num-samples", "-n", type=int, default=50,
                        help="Number of sample NIfTI files to generate")
    parser.add_argument("--shape", "-s", type=int, nargs=3, metavar=("X", "Y", "Z"),
                        default=(128, 128, 128),
                        help="Volume shape as three integers, e.g. -s 128 128 128")
    args = parser.parse_args()

    generate_sample_nifti(args.output_dir, num_samples=args.num_samples, fixed_shape=tuple(args.shape))


if __name__ == "__main__":
    main()
PYEOF

chmod +x chapter2_data_efficiency/generate_sample_data.py

print_success "Created: generate_sample_data.py"
echo ""

print_info "Generating sample data (this may take 1-2 minutes)..."
print_command "python3 chapter2_data_efficiency/generate_sample_data.py --num-samples 200 --shape 128 128 128"
echo ""


print_success "Sample data generated!"
echo ""

print_info "Now creating the caching benchmark script..."

cat > chapter2_data_efficiency/benchmark_caching.py << 'PYEOF'
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

PYEOF

chmod +x chapter2_data_efficiency/benchmark_caching.py

print_success "Created: benchmark_caching.py"
echo ""

print_section "Exercise 11: Ready to Run"

print_warning "⚠️  Run this in a SEPARATE terminal:"
echo ""
print_command "cd $(pwd)"
print_command "python3 chapter2_data_efficiency/benchmark_caching.py --cache-rates 1.0,0.2 --epochs 4"
echo ""

cat << 'EOF'
⏱️  ESTIMATED TIME: 5-8 minutes

This benchmark will:
  1. Test 4 caching strategies
  2. Run 3 epochs for each
  3. Measure time and memory usage
  4. Generate comparison plots

WHAT TO WATCH:
  • First epoch of CacheDataset will be slow (loading data)
  • Subsequent epochs will be MUCH faster
  • SmartCacheDataset adapts over time

EXPECTED SPEEDUPS:
  • CacheDataset (100%): 8-12× faster
  • CacheDataset (50%): 4-6× faster
  • SmartCacheDataset: 5-8× faster
EOF

pause

# ================================================================
# EXERCISE 12: PERSISTENT CACHING
# ================================================================

print_header "EXERCISE 12: Persistent Caching"

cat << 'EOF'
📚 CONCEPT: Persistent Caching

PROBLEM WITH RAM CACHING:
  • Lost when training ends or crashes
  • Must reload every session
  • Doesn't help for very large datasets (> RAM)

SOLUTION: Persistent Cache
  • Pre-process data ONCE
  • Save to fast disk (SSD/NVMe)
  • Load pre-processed data in future sessions
  • 10× faster than loading raw files
  • Survives crashes and restarts

WORKFLOW:

  First Training Session:
  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ Raw Data │ →  │ Process  │ →  │  Cache   │ →  │ Training │
  │ (NIfTI)  │    │ Pipeline │    │  to SSD  │    │          │
  └──────────┘    └──────────┘    └──────────┘    └──────────┘
      Slow            Slow           One-time         Fast

  Subsequent Sessions:
  ┌──────────┐                     ┌──────────┐    ┌──────────┐
  │   Raw    │                     │   Load   │ →  │ Training │
  │   Data   │    (SKIP!)          │   Cache  │    │          │
  └──────────┘                     └──────────┘    └──────────┘
                                       Fast!          Fast!

BENEFITS:
  ✓ Resume training instantly
  ✓ Survive crashes/restarts
  ✓ Share preprocessed data with team
  ✓ Works with unlimited dataset size

USE CASES:
  • Multi-day training runs
  • Expensive preprocessing (registration, augmentation)
  • Distributed training (shared cache)
  • Experimentation (try different models on same data)
EOF

pause

print_section "Creating Exercise 12 Script"

cat > chapter2_data_efficiency/persistent_caching_demo.py << 'PYEOF'
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


PYEOF

chmod +x chapter2_data_efficiency/persistent_caching_demo.py

print_success "Created: persistent_caching_demo.py"
echo ""

print_section "Exercise 12: Ready to Run"

print_warning "⚠️  Run this in a SEPARATE terminal:"
echo ""
print_command "cd $(pwd)"
print_command "python3 chapter2_data_efficiency/persistent_caching_demo.py"
echo ""

cat << 'EOF'
⏱️  ESTIMATED TIME: 3-5 minutes

This demo will:
  1. First run: Create persistent cache (~2 min)
  2. Resumed run: Load from cache (~30 sec)
  3. Show dramatic speedup!

INTERACTIVE:
  • Script will pause between runs
  • Press ENTER to simulate resuming training
  • Compare creation times!

EXPECTED RESULTS:
  First run:   60-120s (creating cache)
  Resumed run: 5-10s (loading cache)
  Speedup:     10-20× faster! 🚀
EOF

pause


# ================================================================
# EXERCISE 13: DATALOADER OPTIMIZATION
# ================================================================

print_header "EXERCISE 13: DataLoader Optimization"

cat << 'EOF'
📚 CONCEPT: DataLoader num_workers and Prefetching

PROBLEM: Single-threaded data loading is slow

  Timeline with num_workers=0 (single thread):
  ┌────────────────────────────────────────────────────┐
  │ CPU: Load batch 1 → GPU trains → Load batch 2 ... │
  │ GPU: ░░░░░░░░░░░░░░ ████████ ░░░░░░░░░░░ ████████ │
  │                     ↑ idle!              ↑ idle!  │
  └────────────────────────────────────────────────────┘

SOLUTION: Multi-worker parallel loading + prefetching

  Timeline with num_workers=4 + prefetch_factor=2:
  ┌────────────────────────────────────────────────────┐
  │ Workers: Batch 2,3 ready │ Batch 3,4 ready │ ...  │
  │ GPU:     ████████████████ ████████████████ ...     │
  │          ↑ always busy!                            │
  └────────────────────────────────────────────────────┘

KEY PARAMETERS:

1. num_workers: Number of parallel data loading processes
   • 0: Single-threaded (slow, but simple)
   • 2-4: Good for most cases
   • 4-8: High-end workstations
   • Too many: Overhead, memory issues

   Rule of thumb: num_workers = min(4, num_cpu_cores // 2)

2. prefetch_factor: How many batches to prepare ahead
   • Default: 2 (prepare 2 batches per worker)
   • Higher: More memory, better GPU utilization
   • Lower: Less memory, possible GPU starvation

   Rule of thumb: prefetch_factor = 2-4

3. pin_memory: Faster CPU→GPU transfer
   • True: Allocate page-locked memory (faster transfer)
   • False: Regular memory (saves RAM)

   Rule of thumb: Always True if you have enough RAM

4. persistent_workers: Keep workers alive between epochs
   • True: Faster epoch transitions (no worker respawn)
   • False: Lower memory during validation/testing

   Rule of thumb: True for training, False for inference

MEDICAL IMAGING CONSIDERATIONS:
  • Large files (100-500 MB): Fewer workers (2-4)
  • Small files (< 10 MB): More workers (4-8)
  • Disk I/O bound: More workers doesn't always help
  • RAM limited: Reduce prefetch_factor

EXAMPLE CONFIGURATIONS:

Workstation (32GB RAM, 8 cores, SSD):
  num_workers=4, prefetch_factor=2, pin_memory=True

Server (128GB RAM, 32 cores, NVMe):
  num_workers=8, prefetch_factor=4, pin_memory=True

Laptop (16GB RAM, 4 cores):
  num_workers=2, prefetch_factor=2, pin_memory=False
EOF

pause

print_section "Creating Exercise 13 Script"

cat > chapter2_data_efficiency/dataloader_optimization.py << 'PYEOF'
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
PYEOF

chmod +x chapter2_data_efficiency/dataloader_optimization.py

print_success "Created: dataloader_optimization.py"
echo ""

print_section "Exercise 13: Ready to Run"
print_warning "⚠️  Run this in a SEPARATE terminal:"
echo ""
print_command "cd $(pwd)"
print_command "python3 chapter2_data_efficiency/dataloader_optimization.py"
echo ""

# ================================================================
# EXERCISE 14: PERSISTENT WORKERS
# ================================================================

print_header "EXERCISE 14: Persistent Workers"

cat << 'EOF'
📚 CONCEPT: Persistent Workers

PROBLEM:
  • By default, PyTorch DataLoader respawns worker processes at the start of every epoch.
  • This adds unnecessary overhead when your dataset is large or you have expensive augmentations.

SOLUTION:
  • persistent_workers=True keeps the same worker processes alive across epochs.

TIMELINE COMPARISON:

Without persistent_workers (default):
  ┌──────────────────────────────────────────────┐
  │ Epoch 1 │ spawn 4 workers │ train │ stop... │
  │ Epoch 2 │ spawn 4 workers │ train │ stop... │
  └──────────────────────────────────────────────┘
  ❌ Extra cost each epoch (seconds wasted)

With persistent_workers=True:
  ┌──────────────────────────────────────────────┐
  │ Spawn once → Reuse → Reuse → Reuse ...       │
  └──────────────────────────────────────────────┘
  ✅ 1× cost only at first epoch, faster transitions

BEST PRACTICE:
  • Use persistent_workers=True for long multi-epoch training.
  • Disable it (False) for validation/inference to free memory sooner.

This exercise measures that overhead and shows how persistent_workers
keeps RAM slightly higher but removes restart time between epochs.
EOF
pause


# ================================================================
# EXERCISE 14: PREFETCH + PERSISTENT INTERACTION
# ================================================================

print_header "EXERCISE 14b: Prefetch + Persistent Interaction"

cat << 'EOF'
📚 CONCEPT: Combining Prefetching with Persistent Workers

GOAL: Measure synergy between
  • prefetch_factor (number of batches prepared ahead)
  • persistent_workers (no respawn between epochs)

PATTERN:
  prefetch_factor ↑ → smoother GPU utilization
  persistent_workers=True → removes worker restart penalty

When both are active, the DataLoader achieves near-continuous
GPU feeding with minimal CPU wait time.

We will visualize throughput and RAM usage for multiple combinations.
EOF

pause

print_section "Creating Exercise 14b Script"

cat > chapter2_data_efficiency/dataloader_persistent_prefetch.py << 'PYEOF'
#!/usr/bin/env python3
"""
Exercise 14b: Prefetch + Persistent Interaction
------------------------------------------------
Benchmark combined effects of persistent_workers and prefetch_factor.
"""

import itertools, time, torch, psutil, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityRanged, ToTensord,
)

def get_transforms():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(keys=["image"], a_min=-200,a_max=200,b_min=0,b_max=1,clip=True),
        ToTensord(keys=["image"]),
    ])

def bench(ds,nw,pin,persist,pref,num_epochs=2):
    process=psutil.Process()
    mem0=process.memory_info().rss/1024**2
    kwargs=dict(dataset=ds,batch_size=4,shuffle=True,num_workers=nw,
                pin_memory=pin,persistent_workers=persist)
    if nw>0: kwargs["prefetch_factor"]=pref
    dl=DataLoader(**kwargs)
    t0=time.time()
    for e in range(num_epochs):
        for b in dl: _=b["image"].mean()
    t=time.time()-t0
    mem1=process.memory_info().rss/1024**2
    thr=len(dl)*4*num_epochs/t
    return dict(name=f"{nw}w pin={pin} pers={persist} pre={pref}",
                t=t/num_epochs,thr=thr,mem=mem1-mem0)

def main():
    data_dir=Path("chapter2_data_efficiency/data/nifti_samples")
    imgs=sorted(data_dir.glob("*.nii.gz"))[:40]
    data=[{"image":str(f)} for f in imgs]
    ds=CacheDataset(data,transform=get_transforms(),cache_rate=1.0,num_workers=4)
    results=[]
    for nw,pin,persist,pref in itertools.product([0,2,4],[False,True],[False,True],[1,2,4]):
        if nw==0 and pref!=2: continue
        results.append(bench(ds,nw,pin,persist,pref))
    base=[r for r in results if r["name"].startswith("0w")][0]

    Path("chapter2_data_efficiency/figures").mkdir(parents=True, exist_ok=True)
    names=[r["name"] for r in results]; speed=[base["t"]/r["t"] for r in results]
    plt.figure(figsize=(12,5))
    plt.bar(names,speed,color="cornflowerblue")
    plt.xticks(rotation=75,ha="right");plt.ylabel("Speedup × vs baseline")
    plt.title("Prefetch + Persistent Interaction")
    plt.tight_layout()
    plt.savefig("chapter2_data_efficiency/figures/prefetch_persistent_interaction.png",dpi=150)
    print("✓ Saved: figures/prefetch_persistent_interaction.png")
    best=max(results,key=lambda r:r["thr"])
    print(f"\n🏆 Best: {best['name']}  ({best['thr']:.1f} samples/s, {base['t']/best['t']:.2f}× faster)")
    print("  Memory Δ: %+0.0f MB" % best["mem"])

if __name__=="__main__":
    main()
PYEOF

chmod +x chapter2_data_efficiency/dataloader_persistent_prefetch.py
print_success "Created: dataloader_persistent_prefetch.py"
echo ""

print_section "Exercise 14b: Ready to Run"
print_command "python3 chapter2_data_efficiency/dataloader_persistent_prefetch.py"
cat << 'EOF'
⏱️  ESTIMATED TIME: 8–12 minutes
This script:
  1. Creates cached dataset
  2. Tests combinations of num_workers, pin_memory, persistent_workers, prefetch_factor
  3. Outputs comparative plots

WHAT TO EXPECT:
  • baseline (num_workers=0) is slowest
  • persistent_workers + prefetch=4 gives highest throughput
  • small RAM increase for best configurations
EOF

pause






cat << 'EOF'
⏱️  ESTIMATED TIME: 8-12 minutes

This benchmark will:
  1. Create cached dataset (2 min)
  2. Test 7 DataLoader configurations (6-8 min)
  3. Generate performance plots

WHAT TO WATCH:
  • num_workers=0 is slowest (baseline)
  • More workers = better GPU utilization (up to a point)
  • pin_memory=True gives ~10% speedup
  • persistent_workers=True speeds up epoch transitions

MONITOR IN ANOTHER TERMINAL:
  watch -n 1 "ps aux | grep python | wc -l"
  → You'll see worker processes spawn!

EXPECTED SPEEDUP:
  • num_workers=4: 2-4× faster than baseline
  • + pin_memory: Additional 10-15% faster
  • + prefetch_factor=4: Additional 5-10% faster
EOF

pause


