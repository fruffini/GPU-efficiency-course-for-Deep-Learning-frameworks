#!/bin/bash

# ================================================================
# CHAPTER 1 — IMPROVE MODEL PERFORMANCE
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

pause_with_message() {
    echo ""
    echo -e "${YELLOW}$1${NC}"
    read -p "Press ENTER when ready..."
    echo ""
}

# ================================================================
# WELCOME SCREEN
# ================================================================

clear
print_header "CHAPTER 1: IMPROVE MODEL PERFORMANCE"

cat << 'EOF'
    ____  ____  __  __   ____        __  _           _          __  _
   / ___||  _ \|  \/  | / __ \      / / | |_(_)_ __ (_)______ _| |_(_) ___  _ __
  | |  _ | |_) | |\/| || |  | |    / /  | __| | '_ \| |_  / _` | __| |/ _ \| '_ \
  | |_| ||  __/| |  | || |__| |   / /   | |_| | | | | |/ / (_| | |_| | (_) | | | |
   \____||_|   |_|  |_(_)____/   /_/     \__|_|_| |_|_/___\__,_|\__|_|\___/|_| |_|

EOF

echo ""
echo -e "${CYAN}Welcome to Chapter 1: Improve Model Performance${NC}"
echo ""
echo "In this chapter, you will learn:"
echo "  • How to optimize GPU memory usage"
echo "  • Mixed precision training (AMP, BF16, FP16)"
echo "  • Batch size and volume scaling strategies"
echo "  • Gradient accumulation techniques"
echo "  • FLOPS calculation and efficiency metrics"
echo "  • How to maximize GPU utilization"
echo ""
echo "Prerequisites:"
echo "  ✓ Completed Chapter 0 (GPU monitoring)"
echo "  ✓ PyTorch with CUDA installed"
echo "  ✓ MONAI library installed"
echo "  ✓ GPU with at least 16GB memory"
echo ""
echo -e "${YELLOW}IMPORTANT: Each exercise script will be generated here,${NC}"
echo -e "${YELLOW}but you must run them in a SEPARATE terminal window!${NC}"
echo ""

pause

# ================================================================
# SETUP CHECK
# ================================================================

print_header "STEP 0: Environment Setup Check"

print_section "Checking Python Environment"

print_info "Checking Python version..."
python3 --version

print_info "Checking CUDA availability..."
python3 << 'PYEOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU Memory: {mem_gb:.1f} GB")
else:
    print("ERROR: CUDA not available!")
    exit(1)
PYEOF

print_success "CUDA environment OK"
echo ""

print_info "Checking required packages..."
python3 << 'PYEOF'
import sys
packages = {'torch': 'PyTorch', 'monai': 'MONAI', 'matplotlib': 'Matplotlib'}
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
PYEOF

print_success "All packages installed"
echo ""

print_section "Creating Directory Structure"

mkdir -p chapter1_improve_the_model_performance/figures
mkdir -p chapter1_improve_the_model_performance/results
mkdir -p chapter1_improve_the_model_performance/logs

print_success "Directories created"
print_info "  • chapter1_improve_the_model_performance/figures/ - for plots"
print_info "  • chapter1_improve_the_model_performance/results/ - for benchmark results"
print_info "  • chapter1_improve_the_model_performance/logs/ - for training logs"

pause

# ================================================================
# EXERCISE 5: BASIC OPTIMIZATION STRATEGIES
# ================================================================

print_header "EXERCISE 5: Basic Optimization Strategies"

cat << 'EOF'
📚 CONCEPT: Basic GPU Optimization

Medical imaging models often underutilize GPUs due to:
  • Small batch sizes (limited by memory)
  • Inefficient precision (FP32 uses 4 bytes per number)
  • Suboptimal optimizer choices

KEY OPTIMIZATIONS:
  1. Batch Size: Larger batches → better GPU utilization
  2. Mixed Precision (AMP): Use FP16 where safe → 2× faster
  3. Optimizer Choice: Adam (faster) vs SGD (less memory)

EXAMPLE:
  Baseline:  batch=1, FP32 → 2.1 samples/s, 6.2 GB
  Optimized: batch=4, AMP  → 6.3 samples/s, 8.1 GB

Result: 3× faster with only 30% more memory!
EOF

pause

print_section "Creating Exercise 5 Script"

print_info "Generating mock_training_opt.py..."

cat > chapter1_improve_the_model_performance/mock_training_opt.py << 'PYEOF'
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
PYEOF

chmod +x chapter1_improve_the_model_performance/mock_training_opt.py

print_success "Script created: mock_training_opt.py"
echo ""

print_section "Instructions to Run Exercise 5"

cat << 'EOF'
📋 STEP-BY-STEP INSTRUCTIONS:

1. Open a NEW terminal window (separate from this one)

2. Navigate to this directory:
   cd <your-current-directory>

3. Activate your Python environment:
   source /path/to/your/venv/bin/activate

4. Run the BASELINE configuration:
EOF

print_command "python3 chapter1_improve_the_model_performance/mock_training_opt.py --batch_size 2 --vol_size 96"

echo ""
echo "5. Observe the output: memory usage, time per epoch, loss values"
echo ""
echo "6. Then run the OPTIMIZED configuration with AMP:"
echo ""

print_command "python3 chapter1_improve_the_model_performance/mock_training_opt.py --batch_size 2 --vol_size 96 --amp"

echo ""
echo "7. Compare the results:"
echo "   • Memory should be ~30-40% lower"
echo "   • Training should be ~1.5-2× faster"
echo ""

cat << 'EOF'
💡 ADDITIONAL EXPERIMENTS (Optional):

Try these configurations in your separate terminal:

A. Larger batch:
   python3 chapter1_improve_the_model_performance/mock_training_opt.py --batch_size 4 --amp

B. Different optimizer:
   python3 chapter1_improve_the_model_performance/mock_training_opt.py --optimizer sgd --amp

C. Larger volume:
   python3 chapter1_improve_the_model_performance/mock_training_opt.py --vol_size 128 --amp

D. Maximum optimization:
   python3 chapter1_improve_the_model_performance/mock_training_opt.py --batch_size 4 --vol_size 128 --amp
EOF

pause_with_message "After running the experiments in your other terminal, press ENTER here to continue..."

# ================================================================
# EXERCISE 6: SCALING BENCHMARK
# ================================================================

print_header "EXERCISE 6: Batch Size and Volume Scaling Benchmark"

cat << 'EOF'
📚 CONCEPT: Scaling Laws in Medical Imaging

Memory and compute scale differently:
  • Batch size: MEMORY scales linearly (2× batch = 2× memory)
  • Volume size: MEMORY scales cubically (2× volume = 8× memory!)
  • Throughput: Usually increases with batch size (up to a point)

WHY THIS MATTERS:
  A 192³ CT scan uses 8× more memory than 96³
  But training 8 samples of 96³ is more efficient!

THIS BENCHMARK TESTS:
  • Batch sizes: [1, 2, 4, 8]
  • Volume sizes: [64, 96, 128, 168, 192]
  • Total: 20 configurations
  • Creates 2 plots showing trade-offs

⏱️  This takes 5-10 minutes to complete!
EOF

pause

print_section "Creating Exercise 6 Script"

print_info "Generating mock_training_scaling.py..."

cat > chapter1_improve_the_model_performance/mock_training_scaling.py << 'PYEOF'
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
PYEOF

chmod +x chapter1_improve_the_model_performance/mock_training_scaling.py

print_success "Script created: mock_training_scaling.py"
echo ""

print_section "Instructions to Run Exercise 6"

print_warning "⚠️  Some configurations may fail with OOM (Out Of Memory) - this is expected!"
echo ""

cat << 'EOF'
📋 RUN THIS IN YOUR SEPARATE TERMINAL:
EOF

print_command "python3 chapter1_improve_the_model_performance/mock_training_scaling.py"

echo ""
cat << 'EOF'
This will:
  • Test 20 different configurations
  • Take approximately 5-10 minutes
  • Generate 2 plots in the figures/ directory
  • Show optimal configuration at the end

📊 AFTER COMPLETION, VIEW THE PLOTS:
  • figures/memory_scaling.png - Shows memory growth
  • figures/throughput_scaling.png - Shows efficiency

To view plots (if you have eog):
  eog chapter1_improve_the_model_performance/figures/memory_scaling.png &
  eog chapter1_improve_the_model_performance/figures/throughput_scaling.png &
EOF

pause_with_message "After the benchmark completes in your other terminal, press ENTER to continue..."

print_info "Interpretation tips:"
echo ""
echo "Memory Plot:"
echo "  • Memory grows ~cubically with volume size"
echo "  • Doubling volume (96→192) = 8× memory!"
echo "  • Each line = one batch size"
echo ""
echo "Throughput Plot:"
echo "  • Higher = better efficiency"
echo "  • Usually peaks at batch size 4-8"
echo "  • Very large volumes may slow down"
echo ""

pause

# ================================================================
# EXERCISE 7: PRECISION COMPARISON
# ================================================================

print_header "EXERCISE 7: Precision Comparison Benchmark"

cat << 'EOF'
📚 CONCEPT: Numerical Precision Trade-offs

FLOATING POINT FORMATS:
  • FP32 (32-bit): 4 bytes, standard, highest accuracy
  • FP16 (16-bit): 2 bytes, 2× faster, less stable
  • BF16 (16-bit): 2 bytes, 2× faster, more stable than FP16
  • AMP (Mixed): Automatically chooses FP16 or FP32 per operation

PRECISION HIERARCHY:
  FP32 > BF16 ≈ AMP > FP16  (stability)
  FP16 > BF16 ≈ AMP > FP32  (speed)

GPU SUPPORT:
  • V100 (Volta):    FP32, FP16, AMP
  • A100 (Ampere):   FP32, FP16, BF16, AMP  ← Best for BF16
  • H100 (Hopper):   FP32, FP16, BF16, AMP, FP8

RECOMMENDATION:
  Use AMP (Automatic Mixed Precision) - best balance!
EOF

pause

print_section "Creating Exercise 7 Script"

# [Script content here - same as before]
cat > chapter1_improve_the_model_performance/mock_precision_benchmark.py << 'PYEOF'
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
PYEOF

chmod +x chapter1_improve_the_model_performance/mock_precision_benchmark.py

print_success "Script created: mock_precision_benchmark.py"
echo ""

print_section "Instructions to Run Exercise 7"

cat << 'EOF'
📋 RUN THIS IN YOUR SEPARATE TERMINAL:
EOF

print_command "python3 chapter1_improve_the_model_performance/mock_precision_benchmark.py"

echo ""
echo "This will test all supported precisions on your GPU."
echo "Results saved to figures/precision_memory.png and figures/precision_throughput.png"
echo ""

pause_with_message "After the benchmark completes, press ENTER to continue..."

cat << 'EOF'

📊 TYPICAL RESULTS:

Precision  | Memory Reduction | Speedup  | Stability
-----------|------------------|----------|----------
FP32       | Baseline         | 1.0×     | ★★★★★
AMP        | ~40%             | ~1.8×    | ★★★★★
BF16       | ~45%             | ~2.1×    | ★★★★★
FP16       | ~50%             | ~2.3×    | ★★★☆☆

RECOMMENDATION:
  → Use AMP for production training
  → Use BF16 if you have A100/H100
  → Avoid pure FP16 (unstable for medical imaging)
EOF

pause

# ================================================================
# EXERCISE 8: GRADIENT ACCUMULATION
# ================================================================

print_header "EXERCISE 8: Gradient Accumulation + Precision"

cat << 'EOF'
📚 CONCEPT: Gradient Accumulation

PROBLEM:
  Large batch sizes don't fit in GPU memory
  Example: Want batch=16, but only batch=2 fits

SOLUTION: Gradient Accumulation
  1. Forward + backward with small batch (2)
  2. Accumulate gradients (don't update weights yet)
  3. Repeat N times (e.g., 8 times)
  4. Update weights once (effective batch = 2×8 = 16)

BENEFITS:
  ✓ Simulate large batches without OOM
  ✓ Memory stays constant
  ✓ Training stability improves

TRADE-OFFS:
  ✗ Slightly slower (more forward/backward passes)
  ✗ No batch norm benefits within micro-batches

⏱️  This benchmark takes 10-15 minutes!
EOF

pause

print_section "Creating Exercise 8 Script"

cat > chapter1_improve_the_model_performance/mock_precision_accumulation.py << 'PYEOF'
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

PYEOF
chmod +x chapter1_improve_the_model_performance/mock_precision_accumulation.py

print_success "Script created: mock_precision_accumulation.py"
echo ""

print_section "Instructions to Run Exercise 8"

cat << 'EOF'
📋 RUN THIS IN YOUR SEPARATE TERMINAL:
EOF

print_command "python3 chapter1_improve_the_model_performance/mock_precision_accumulation.py"

echo ""
echo "Testing 15 configurations (3 precisions × 5 accumulation steps)"
echo "This will take approximately 10-15 minutes."
echo ""

pause_with_message "After the benchmark completes, press ENTER to continue..."

cat << 'EOF'

📊 KEY FINDINGS & TECHNICAL EXPLANATION
────────────────────────────────────────────

1️⃣ MEMORY STAYS CONSTANT
   Gradient accumulation does **not** increase memory consumption!
   When you use `accum_steps > 1`, the model processes several *micro-batches*
   **sequentially**, not in parallel.

   🔍 For each forward/backward pass:
   - Only the activations for the **current micro-batch** are kept in GPU memory.
   - After `.backward()`, those activations are **freed immediately**.
   - Gradients are **added in place** to the parameter `.grad` buffers, without duplication.

   This means:
batch=2, accum=1 → GPU holds activations for 2 samples
batch=2, accum=8 → GPU still holds activations for 2 samples (reused 8 times)

sql
Copy code
🧠 Memory stays constant because you’re simulating a larger batch **in time**, not **in space**.

The only tiny variations (± few MB) may come from:
- CUDA allocator fragmentation
- AMP temporary buffers
- FP32 gradient storage for mixed precision

────────────────────────────────────────────

2️⃣ THROUGHPUT SLIGHTLY DECREASES
Since you perform more forward/backward passes per optimizer step,
the total wall time increases linearly with the accumulation factor.
However, the stability of gradient estimates improves dramatically
— reducing noise and improving convergence for small batches.

────────────────────────────────────────────

3️⃣ BEST COMBINATION
For most 3D medical imaging workloads:
AMP + accum=4 → Best trade-off of speed, memory, and stability
BF16 + accum=4 → Excellent for A100/H100 GPUs

yaml
Copy code

🏁 Result:
- Memory: constant across accumulation steps
- Throughput: slightly lower
- Convergence: smoother and more stable

────────────────────────────────────────────
EOF

pause
# ================================================================
# EXERCISE 9: FLOPS AND EFFICIENCY ANALYSIS
# ================================================================

print_header "EXERCISE 9: FLOPS and Efficiency Analysis"

cat << 'EOF'
📚 CONCEPT: Understanding GPU Efficiency

FLOPS = Floating-Point Operations Per Second
  • Measures computational throughput
  • Higher = better GPU utilization
  • Medical imaging: aim for >50% efficiency

WHY FLOPS MATTER:
  GPU A100: 19.5 TFLOPS (FP32) peak capacity

  Scenario 1: Achieve 5 TFLOPS  → 26% efficiency (poor)
  Scenario 2: Achieve 12 TFLOPS → 62% efficiency (good!)
  Scenario 3: Achieve 16 TFLOPS → 82% efficiency (excellent!)

WHAT AFFECTS EFFICIENCY:
  ✓ Batch size (larger = better)
  ✓ Precision (FP16 = 2× theoretical peak)
  ✓ Data loading (if slow, GPU idles)
  ✓ Model architecture (some ops are inefficient)

THIS BENCHMARK:
  • Calculates theoretical model FLOPS
  • Measures actual achieved FLOPS
  • Computes efficiency = achieved / peak × 100%
  • Tests multiple batch sizes and precisions
EOF

pause

print_section "Creating Exercise 9 Script"

print_info "Generating compute_flops_efficiency.py..."

cat > chapter1_improve_the_model_performance/compute_flops_efficiency.py << 'PYEOF'
#!/usr/bin/env python3
"""
Exercise 9: FLOPS and Efficiency Analysis
Compute theoretical and achieved FLOPS, measure GPU efficiency
"""

import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from monai.networks.nets import BasicUNet


def count_conv3d_flops(in_channels, out_channels, kernel_size, input_shape):
    """Count FLOPs for a 3D convolution"""
    D, H, W = input_shape
    K_d, K_h, K_w = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3

    # FLOPs per output position: 2 ops (multiply + add) per kernel element
    flops_per_position = 2 * in_channels * K_d * K_h * K_w
    # Total output positions
    output_positions = D * H * W * out_channels

    return flops_per_position * output_positions


def estimate_model_flops(model, input_shape=(1, 1, 96, 96, 96)):
    """Estimate total FLOPs for BasicUNet"""
    batch, channels, D, H, W = input_shape

    # BasicUNet architecture: (16, 32, 64, 128, 256, 32)
    features = [16, 32, 64, 128, 256]

    total_flops = 0
    current_shape = (D, H, W)
    in_ch = channels

    # Encoder
    for feat in features:
        # Two conv3d per level
        total_flops += 2 * count_conv3d_flops(in_ch, feat, 3, current_shape)
        in_ch = feat
        # Downsample
        current_shape = tuple(s // 2 for s in current_shape)

    # Bottleneck
    total_flops += 2 * count_conv3d_flops(features[-1], features[-1], 3, current_shape)

    # Decoder
    for i in range(len(features) - 1, -1, -1):
        current_shape = tuple(s * 2 for s in current_shape)
        feat = features[i]
        # Upconv + two conv3d
        total_flops += count_conv3d_flops(in_ch, feat, 3, current_shape)
        total_flops += 2 * count_conv3d_flops(feat * 2, feat, 3, current_shape)
        in_ch = feat

    # Final conv
    total_flops += count_conv3d_flops(in_ch, 1, 1, current_shape)

    return total_flops * batch


def get_gpu_peak_flops():
    """Get theoretical peak FLOPS for current GPU"""
    device_name = torch.cuda.get_device_name(0).upper()

    # Peak TFLOPS for common GPUs (FP32)
    peak_flops_map = {
        "A100": 19.5e12,      # 19.5 TFLOPS FP32
        "V100": 15.7e12,      # 15.7 TFLOPS FP32
        "A6000": 38.7e12,     # 38.7 TFLOPS FP32
        "L40S": 91.6e12,      # 91.6 TFLOPS FP32 (with sparsity)
        "L40": 90.5e12,       # 90.5 TFLOPS FP32
        "H100": 60.0e12,      # 60 TFLOPS FP32
        "RTX 4090": 82.6e12,  # 82.6 TFLOPS FP32
        "RTX 3090": 35.6e12,  # 35.6 TFLOPS FP32
        "T4": 8.1e12,         # 8.1 TFLOPS FP32
    }

    for key, flops in peak_flops_map.items():
        if key in device_name:
            return flops

    # Default estimate
    return 20e12


def benchmark_flops(batch_size=2, vol_size=96, precision="fp32", iterations=20):
    """Benchmark model and compute achieved FLOPS"""

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    model = BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        features=(16, 32, 64, 128, 256, 32)
    ).to(device)

    if precision == "fp16":
        model = model.half()

    # Configure precision
    use_amp = precision in ["amp", "bf16"]
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    dtype = torch.float16 if precision == "fp16" else torch.float32
    x = torch.randn(batch_size, 1, vol_size, vol_size, vol_size,
                   device=device, dtype=dtype)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(iterations):
        try:
            torch.cuda.synchronize()
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype):
                _ = model(x)

        except torch.OutOfMemoryError:
            print(f"OOM at batch size {batch_size}, volume size {vol_size}, precision {precision}")
            return {
                "precision": precision,
                "batch_size": batch_size,
                "vol_size": vol_size,
                "time_per_iter": float('inf'),
                "achieved_tflops": 0.0,
                "peak_tflops": 0.0,
                "efficiency_pct": 0.0,
                "memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "samples_per_sec": 0.0
            }
    elapsed = time.time() - start

    # Calculate FLOPS
    model_flops = estimate_model_flops(model, input_shape=(batch_size, 1, vol_size, vol_size, vol_size))
    total_flops = model_flops * iterations
    achieved_flops = total_flops / elapsed
    achieved_tflops = achieved_flops / 1e12

    # Peak FLOPS
    peak_flops = get_gpu_peak_flops()
    if precision in ["amp", "fp16", "bf16"]:
        peak_flops *= 2  # FP16 has 2× theoretical peak

    peak_tflops = peak_flops / 1e12
    efficiency = (achieved_flops / peak_flops) * 100

    # Memory bandwidth
    mem_used = torch.cuda.max_memory_allocated() / 1024**3

    return {
        "precision": precision,
        "batch_size": batch_size,
        "vol_size": vol_size,
        "time_per_iter": elapsed / iterations,
        "achieved_tflops": achieved_tflops,
        "peak_tflops": peak_tflops,
        "efficiency_pct": efficiency,
        "memory_gb": mem_used,
        "samples_per_sec": batch_size * iterations / elapsed
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FLOPS and Efficiency Analysis")
    parser.add_argument("--volume", type=int, default=96,
                        help="Input volume size (default: 96)")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    print("="*80)
    print("Exercise 9: FLOPS and Efficiency Analysis")
    print("="*80)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    peak_flops_fp32 = get_gpu_peak_flops() / 1e12
    peak_flops_fp16 = (get_gpu_peak_flops() * 2) / 1e12
    print(f"Peak FLOPS (FP32): {peak_flops_fp32:.1f} TFLOPS")
    print(f"Peak FLOPS (FP16): {peak_flops_fp16:.1f} TFLOPS")
    print("="*80 + "\n")


    # Test different configurations
    configs = [
        {"batch_size": 1, "vol_size": args.volume, "precision": "fp32"},
        {"batch_size": 2, "vol_size": args.volume, "precision": "fp32"},
        {"batch_size": 4, "vol_size": args.volume, "precision": "fp32"},
        {"batch_size": 8, "vol_size": args.volume, "precision": "fp32"},
        {"batch_size": 16, "vol_size": args.volume, "precision": "fp32"},
        {"batch_size": 1, "vol_size": args.volume, "precision": "amp"},
        {"batch_size": 2, "vol_size": args.volume, "precision": "amp"},
        {"batch_size": 4, "vol_size": args.volume, "precision": "amp"},
        {"batch_size": 8, "vol_size": args.volume, "precision": "amp"},
        {"batch_size": 16, "vol_size": args.volume, "precision": "amp"},
        {"batch_size": 1, "vol_size": args.volume, "precision": "bf16"},
        {"batch_size": 2, "vol_size": args.volume, "precision": "bf16"},
        {"batch_size": 4, "vol_size": args.volume, "precision": "bf16"},
        {"batch_size": 8, "vol_size": args.volume, "precision": "bf16"},
        {"batch_size": 16, "vol_size": args.volume, "precision": "bf16"},
    ]


    results = []

    print(f"{'Config':<20} {'TFLOPS':<10} {'Peak':<10} {'Efficiency':<12} {'Mem (GB)':<10}")
    print("-"*80)

    for config in configs:
        result = benchmark_flops(**config)
        results.append(result)

        config_str = f"{config['precision'].upper()}, B={config['batch_size']}, V={config['vol_size']}"
        print(f"{config_str:<20} {result['achieved_tflops']:<10.2f} "
              f"{result['peak_tflops']:<10.1f} {result['efficiency_pct']:<12.1f}% "
              f"{result['memory_gb']:<10.2f}")

    print("\n" + "="*80)
    print("Analysis Summary")
    print("="*80 + "\n")

    # Find best efficiency
    best_eff = max(results, key=lambda x: x['efficiency_pct'])
    print(f"🚀 Best GPU Efficiency: {best_eff['efficiency_pct']:.1f}%")
    print(f"   Config: {best_eff['precision'].upper()}, "
          f"Batch={best_eff['batch_size']}, Vol={best_eff['vol_size']}")
    print(f"   Achieved: {best_eff['achieved_tflops']:.2f} TFLOPS")
    print(f"   Peak: {best_eff['peak_tflops']:.1f} TFLOPS")
    print()

    # Find best throughput
    best_throughput = max(results, key=lambda x: x['samples_per_sec'])
    print(f"⚡ Best Throughput: {best_throughput['samples_per_sec']:.2f} samples/s")
    print(f"   Config: {best_throughput['precision'].upper()}, "
          f"Batch={best_throughput['batch_size']}, Vol={best_throughput['vol_size']}")
    print()

    # Memory efficiency
    print("💾 Memory Efficiency:")
    for result in results:
        samples_per_gb = result['samples_per_sec'] / result['memory_gb']
        print(f"   {result['precision'].upper()}, B={result['batch_size']}: "
              f"{samples_per_gb:.2f} samples/s/GB")
    print()

    # Create visualizations
    print("Generating plots...")

    # Plot 1: FLOPS by configuration
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # FLOPS comparison
    ax = axes[0, 0]
    configs_labels = [f"{r['precision'].upper()}\nB={r['batch_size']}" for r in results]
    achieved = [r['achieved_tflops'] for r in results]
    peak = [r['peak_tflops'] for r in results]

    x = np.arange(len(results))
    width = 0.35

    bars1 = ax.bar(x - width/2, achieved, width, label='Achieved',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, peak, width, label='Peak',
                   color='orange', alpha=0.8)

    ax.set_ylabel('TFLOPS', fontsize=11)
    ax.set_title('Achieved vs Peak FLOPS', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs_labels, fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Efficiency comparison
    ax = axes[0, 1]
    efficiency = [r['efficiency_pct'] for r in results]
    colors = ['green' if e > 50 else 'orange' if e > 30 else 'red' for e in efficiency]

    bars = ax.bar(x, efficiency, color=colors, alpha=0.7, edgecolor='black')
    for bar, eff in zip(bars, efficiency):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{eff:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('GPU Utilization (%)', fontsize=11)
    ax.set_title('GPU Efficiency', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs_labels, fontsize=9)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% target')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Throughput comparison
    ax = axes[1, 0]
    throughput = [r['samples_per_sec'] for r in results]
    bars = ax.bar(x, throughput, color='purple', alpha=0.7, edgecolor='black')

    for bar, thr in zip(bars, throughput):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{thr:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Samples per Second', fontsize=11)
    ax.set_title('Training Throughput', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs_labels, fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Memory usage
    ax = axes[1, 1]
    memory = [r['memory_gb'] for r in results]
    bars = ax.bar(x, memory, color='teal', alpha=0.7, edgecolor='black')

    for bar, mem in zip(bars, memory):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mem:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('GPU Memory (GB)', fontsize=11)
    ax.set_title('Memory Usage', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs_labels, fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'./chapter1_improve_the_model_performance/figures/flops_analysis_{args.volume}.png', dpi=150)
    plt.close()

    # Plot 2: Efficiency breakdown
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by precision
    fp32_results = [r for r in results if r['precision'] == 'fp32']
    amp_results = [r for r in results if r['precision'] == 'amp']
    bf16_results = [r for r in results if r['precision'] == 'bf16']

    fp32_batches = [r['batch_size'] for r in fp32_results]
    fp32_eff = [r['efficiency_pct'] for r in fp32_results]

    amp_batches = [r['batch_size'] for r in amp_results]
    amp_eff = [r['efficiency_pct'] for r in amp_results]

    bf16_batches = [r['batch_size'] for r in bf16_results]
    bf16_eff = [r['efficiency_pct'] for r in bf16_results]

    ax.plot(fp32_batches, fp32_eff, '-o', linewidth=2, markersize=10,
            label='FP32', color='steelblue')
    ax.plot(amp_batches, amp_eff, '-s', linewidth=2, markersize=10,
            label='AMP', color='orange')
    ax.plot(bf16_batches, bf16_eff, '-^', linewidth=2, markersize=10,
            label='BF16', color='green')

    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('GPU Efficiency (%)', fontsize=12)
    ax.set_title('GPU Efficiency vs Batch Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% target')

    plt.tight_layout()
    plt.savefig(f'./chapter1_improve_the_model_performance/figures/efficiency_vs_batch_{args.volume}.png', dpi=150)
    plt.close()

    print("✓ Saved: figures/flops_analysis.png")
    print("✓ Saved: figures/efficiency_vs_batch.png")

    print("\n" + "="*80)
    print("Key Insights")
    print("="*80)

    print("""
1. GPU EFFICIENCY:
   • >70% = Excellent (compute-bound, GPU well utilized)
   • 40-70% = Good (some optimization possible)
   • <40% = Poor (CPU bottleneck or small batches)

2. MEMORY vs COMPUTE:
   • Small batches: Memory underutilized, low FLOPS
   • Large batches: Better FLOPS, but may OOM
   • Sweet spot: Batch size that achieves >50% efficiency

3. PRECISION IMPACT:
   • FP32: Lower efficiency, higher memory
   • AMP/FP16: Higher efficiency, lower memory
   • Always use AMP for 3D medical imaging!

4. OPTIMIZATION STRATEGY:
   • Start with small batch, measure efficiency
   • Increase batch until efficiency peaks or OOM
   • Use AMP to double effective compute capacity
   • Monitor: aim for >50% GPU utilization
    """)

    print("="*80)


if __name__ == "__main__":
    main()
PYEOF

chmod +x chapter1_improve_the_model_performance/compute_flops_efficiency.py

print_success "Script created: compute_flops_efficiency.py"
echo ""

print_section "Running Exercise 9 in a Separate Terminal"

print_warning "⚠️  IMPORTANT: You need to run this script in a SEPARATE terminal!"
echo ""
print_info "Open a new terminal and run:"
print_command "cd $(pwd)"
print_command "python3 chapter1_improve_the_model_performance/compute_flops_efficiency.py --volume=96"
echo ""
print_info "This benchmark will:"
print_info "  • Test 6 configurations (takes ~3-5 minutes)"
print_info "  • Measure GPU FLOPS and efficiency"
print_info "  • Generate 2 analysis plots"
echo ""

cat << 'EOF'
📊 WHAT TO WATCH FOR:

While the script runs, open another terminal and monitor GPU:
  watch -n 1 nvidia-smi

You should see:
  • GPU utilization increasing with batch size
  • Memory usage growing with batch size
  • Temperature rising slightly under load

EXPECTED OUTPUT:
  Config               TFLOPS     Peak       Efficiency   Mem (GB)
  --------------------------------------------------------------------
  FP32, B=1, V=96     3.42       19.5       17.5%        2.31
  FP32, B=2, V=96     6.18       19.5       31.7%        3.84
  ...
  AMP, B=8, V=96      31.47      39.0       80.7%        9.16
EOF

echo ""
pause


# ================================================================
# FINAL SUMMARY
# ================================================================

print_header "CHAPTER 1 COMPLETE! 🎉"

cat << 'EOF'
═══════════════════════════════════════════════════════════════
                    📚 WHAT YOU LEARNED
═══════════════════════════════════════════════════════════════

✓ Exercise 5: Basic optimization flags (batch size, AMP, optimizer)
✓ Exercise 6: Scaling laws (memory vs volume, throughput)
✓ Exercise 7: Precision comparison (FP32, AMP, BF16, FP16)
✓ Exercise 8: Gradient accumulation + precision
✓ Exercise 9: FLOPS calculation and GPU efficiency

═══════════════════════════════════════════════════════════════
                   🚀 OPTIMIZATION RECOMMENDATIONS
═══════════════════════════════════════════════════════════════

FOR MOST MEDICAL IMAGING TASKS:
  1. Use AMP (--amp flag)
  2. Batch size = 4-8
  3. Volume size = 96³ or 128³
  4. Gradient accumulation = 4
  5. Optimizer = Adam

EXAMPLE COMMAND:
  python train.py \\
    --batch_size 4 \\
    --vol_size 96 \\
    --amp \\
    --gradient_accumulation_steps 4 \\
    --optimizer adam

EXPECTED IMPROVEMENTS:
  • 2-3× faster training
  • 30-40% less memory
  • Better convergence

═══════════════════════════════════════════════════════════════
                        📊 YOUR RESULTS
═══════════════════════════════════════════════════════════════

All scripts saved in:
  chapter1_improve_the_model_performance/

All plots saved in:
  chapter1_improve_the_model_performance/figures/

  • memory_scaling.png
  • throughput_scaling.png
  • precision_memory.png
  • precision_throughput.png
  • memory_accumulation.png
  • throughput_accumulation.png
  • flops_analysis.png
  • efficiency_vs_batch.png

═══════════════════════════════════════════════════════════════
                         🎯 NEXT STEPS
═══════════════════════════════════════════════════════════════

Chapter 2: Data Loading Optimization (Coming Soon)
  • Efficient data loading pipelines
  • Prefetching and caching
  • Multi-worker data loaders
  • Medical imaging-specific optimizations

EOF

print_info "Run this script again anytime: bash run_chapter1_exercises.sh"
echo ""
print_success "Tutorial complete! All exercise scripts are ready to use."
echo ""

# ================================================================
# END OF SCRIPT
# ================================================================