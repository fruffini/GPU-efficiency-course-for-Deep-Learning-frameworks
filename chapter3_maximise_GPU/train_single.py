#!/usr/bin/env python3
"""
Single Training Process
Basic training with monitoring
"""

import torch
import psutil
import time
import os
from pathlib import Path
from monai.data import SmartCacheDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Resized,
    ScaleIntensityRanged, RandRotate90d, ToTensord
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss


def get_transforms():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(96, 96, 96), mode="trilinear"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-200, a_max=200,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(0, 2)),
        ToTensord(keys=["image"]),
    ])


def train_single_process(
    data_dir="chapter2_data_efficiency/data/nifti_samples",
    epochs=3,
    batch_size=2,
    process_id=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pid = process_id if process_id else os.getpid()
    print(f"\n{'='*70}")
    print(f"Training Process [PID: {pid}]")
    print(f"{'='*70}")

    # Load data
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("   Run: python3 chapter2_data_efficiency/generate_sample_data.py")
        return

    imgs = sorted(data_path.glob("*.nii.gz"))[:40]
    if len(imgs) == 0:
        print(f"❌ No images found in {data_dir}")
        return

    print(f"\n📂 Using {len(imgs)} images")
    data_dicts = [{"image": str(f)} for f in imgs]

    # Create dataset
    print("🔄 Creating SmartCacheDataset...")
    transforms = get_transforms()
    dataset = SmartCacheDataset(
        data=data_dicts,
        transform=transforms,
        cache_num=len(data_dicts) // 2,  # Cache 50%
        num_init_workers=2,
        num_replace_workers=1
    )

    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    # Model
    print("🏗️  Building model...")
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

    # Training
    print(f"\n🚀 Starting training: {epochs} epochs, batch={batch_size}")
    print(f"   Device: {device}")

    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(loader):
            x = batch["image"].to(device, non_blocking=True)
            y = (x > 0.5).float()

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                out = model(x)
                loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(loader)}, "
                      f"Loss: {loss.item():.4f}", end='\r')

        torch.cuda.synchronize()
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(loader)

        print(f"  Epoch {epoch+1}/{epochs} done: "
              f"Loss={avg_loss:.4f}, Time={epoch_time:.2f}s" + " "*30)

    total_time = time.time() - start_time

    # Report statistics
    print(f"\n{'='*70}")
    print("Training Complete")
    print(f"{'='*70}")

    gpu_mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    gpu_mem_reserved_mb = torch.cuda.memory_reserved(device) / 1024**2

    print(f"\n⏱️  Total time: {total_time:.2f}s")
    print(f"📊 Throughput: {len(imgs) * epochs / total_time:.2f} samples/s")
    print(f"\n💾 GPU Memory:")
    print(f"   Allocated: {gpu_mem_mb:.0f} MB")
    print(f"   Reserved: {gpu_mem_reserved_mb:.0f} MB")

    # CPU info
    cpu_percent = psutil.cpu_percent(interval=0.5)
    process = psutil.Process()
    process_mem_mb = process.memory_info().rss / 1024**2

    print(f"\n🧠 CPU:")
    print(f"   Load: {cpu_percent:.1f}%")
    print(f"   Process RAM: {process_mem_mb:.0f} MB")

    print(f"\n{'='*70}")

    return {
        'total_time': total_time,
        'gpu_mem_mb': gpu_mem_mb,
        'cpu_percent': cpu_percent
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--process_id", type=int, default=None)
    args = parser.parse_args()

    train_single_process(
        epochs=args.epochs,
        batch_size=args.batch_size,
        process_id=args.process_id
    )
