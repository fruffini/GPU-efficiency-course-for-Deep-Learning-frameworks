#!/usr/bin/env python3
"""
Exercise 5: Optimizing GPU Utilization in 3D Training
"""

import torch
import time
import argparse
from monai.networks.nets import BasicUNet
from monai.losses import DiceLoss
from monai.data import MetaTensor


parser = argparse.ArgumentParser(description="Optimize GPU usage in 3D MONAI training")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
parser.add_argument("--vol_size", type=int, default=96, help="Volume size")
parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--steps", type=int, default=5)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*70)
print("Exercise 5: GPU Optimization Strategies")
print("="*70)
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Configuration:")
print(f"  • Batch size: {args.batch_size}")
print(f"  • Volume size: {args.vol_size}³")
print(f"  • Optimizer: {args.optimizer.upper()}")
print(f"  • AMP enabled: {args.amp}")
print("="*70 + "\n")

# Model setup
model = BasicUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    features=(16, 32, 64, 128, 256, 32)
).to(device)

criterion = DiceLoss(sigmoid=True)
optimizer = (
    torch.optim.Adam(model.parameters(), lr=1e-4)
    if args.optimizer == "adam"
    else torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
)
scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}\n")

# Training loop
start_time = time.time()

for epoch in range(args.epochs):
    print(f"Epoch {epoch+1}/{args.epochs}")
    print("-" * 50)

    for batch in range(args.steps):
        inputs = MetaTensor(
            torch.randn(args.batch_size, 1, args.vol_size, args.vol_size, args.vol_size, device=device)
        )
        targets = MetaTensor(
            torch.randint(0, 2, (args.batch_size, 1, args.vol_size, args.vol_size, args.vol_size),
                         device=device, dtype=torch.float32)
        )

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if batch % 2 == 0:
            mem_alloc = torch.cuda.memory_allocated() / 1024**2
            mem_reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"  Step {batch+1}/{args.steps} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Alloc: {mem_alloc:.0f} MB | "
                  f"Reserved: {mem_reserved:.0f} MB")

        time.sleep(0.2)

    print()

elapsed = time.time() - start_time

print("="*70)
print("Training Complete!")
print(f"Total time: {elapsed:.2f}s")
print(f"Avg time per epoch: {elapsed/args.epochs:.2f}s")
print(f"Peak memory: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")
print("="*70)
