#!/bin/bash

echo "=========================================="
echo "Chapter 0: Practical Exercises"
echo "=========================================="
echo ""

# Exercise 1
echo "Exercise 1: Identify Your GPU Hardware"
echo "======================================="
echo ""
read -p "Press Enter to run..."
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo "LINK to nvidia: https://developer.nvidia.com/cuda-gpus"
echo "Q: How much memory does each GPU have?"
echo "Q: What is the compute capability?"
read -p "Press Enter to continue..."
echo ""

# Exercise 2
echo "Exercise 2: Baseline GPU State"
echo "==============================="
echo ""
read -p "Press Enter to save baseline..."
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > baseline.csv
cat baseline.csv
echo ""
echo "Saved to baseline.csv"
read -p "Press Enter to continue..."
echo ""

# Exercise 3
echo "Exercise 3: Simulate GPU Load (Adjustable Sleep Time)"
echo "======================================================"
echo ""
# Exercise 3
echo "Exercise 3: Simulate GPU Load (with Memory Analysis)"
echo "====================================================="
echo ""

cat > ./tmp_L0/gpu_load.py << 'EOF'
import torch
import time
import os
os.makedirs("./tmp_L0", exist_ok=True)
# ------------------------------------------------------------------------------------
# Rough approximate breakdown on an A100 40 GB GPU:
#
# Component                          Approx. Memory
# ---------------------------------  ---------------
# Tensor x                           ~9.3 GB
# Tensor y                           ~9.3 GB
# cuBLAS workspace / fragmentation   ~8–10 GB
# Total shown by nvidia-smi          ≈ 27–29 GB
#
# This matches exactly what you’re seeing.
# ------------------------------------------------------------------------------------

print("Starting GPU workload...")
device = torch.device('cuda')
print(f"Using device: {device}")

# Allocate a large tensor
x = torch.randn(20000, 20000, device=device)
print(f"Allocated tensor 'x' uses {x.element_size() * x.nelement() / 1024**2:.2f} MB")

# Fixed sleep time (to simulate load pacing)
sleep_time = 0.1

# Print helper for memory diagnostics
def print_memory(prefix=""):
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    fragmentation = reserved - allocated
    print(f"{prefix} Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Fragmentation: {fragmentation:.2f} GB")

print("Running matrix multiplications with periodic memory cleanup...\n")

for i in range(50):  # fewer iterations for safety
    y = torch.mm(x, x)

    if i % 10 == 0:
        print(f"Iteration {i}/50:")
        print_memory("   Before cleanup ->")

        # Demonstrate how to free memory manually
        del y
        torch.cuda.empty_cache()
        print_memory("   After cleanup  ->")
        print()

    time.sleep(sleep_time)

print("\n✅ Workload complete!")
EOF


echo "Terminal 1: python ./tmp_L0/gpu_load.py --sleep 0.1"
echo "Terminal 2: watch -n 1 nvidia-smi || nvtop (remember to load the module : module load nvtop/3.2.0-GCCcore-13.3.0)"
echo ""
read -p "Press Enter to continue..."

# --- GPU memory breakdown explanation ---
echo "--------------------------------------------------------------"
echo "📊 Rough approximate breakdown on an A100 40 GB GPU:"
echo ""
echo "Component                          Approx. Memory"
echo "---------------------------------  ---------------"
echo "Tensor x                                  ~1.5 GB"
echo "Tensor y (mm)                                 ~1.5 GB"
echo "cuBLAS workspace / fragmentation          ~1.5 GB"
echo "Total shown by nvidia-smi || nvtop        ≈ 4.5 GB"
echo ""
echo "This matches exactly what you’re seeing."
echo "--------------------------------------------------------------"
echo ""

# --- cuBLAS explanatory hint ---
echo "--------------------------------------------------------------"
echo "💡 HINT — Understanding cuBLAS and Extra Memory Usage"
echo ""
echo "When you do large matrix multiplications, like:"
echo "  y = torch.mm(x, x)"
echo ""
echo "cuBLAS (NVIDIA’s GPU-accelerated BLAS library) often allocates"
echo "temporary workspace buffers to optimize performance — these are"
echo "extra regions of GPU memory used internally for:"
echo "  • Blocked or tiled matrix multiplication"
echo "  • Intermediate accumulation"
echo "  • Tensor core alignment"
echo "  • Stream synchronization"
echo ""
echo "That’s why GPU memory usage sometimes jumps from the"
echo "theoretical tensor size (~1.5 GB for one 50k×50k tensor) to"
echo "something much higher (~4.5 GB total):"
echo "→ 1/3 of that usage comes from cuBLAS workspaces"
echo "and PyTorch’s caching allocator."
echo "--------------------------------------------------------------"
echo ""

read -p "Press Enter to continue..."

echo ""


# Exercise 4
echo "Exercise 4: Detect Memory Leaks"
echo "================================"
echo ""

# --- Explanatory section (as echo text) ---
echo "📘 This exercise demonstrates how GPU memory can remain occupied"
echo "even after a script finishes allocating tensors — if we keep references"
echo "to them in memory."
echo ""
echo "In this example, we'll allocate several large CUDA tensors and store"
echo "them in a list called 'leaked_tensors'."
echo ""
echo "As long as the list keeps those references, PyTorch cannot free them."
echo "You will see GPU memory usage growing with each allocation."
echo ""
echo "After running the script, check the memory consumption with:"
echo "  nvidia-smi"
echo ""
echo "Only after the script exits (and Python's garbage collector releases"
echo "the list) will the memory be freed."
echo "--------------------------------------------------------------"
echo ""

# --- Create the Python script ---
cat > ./tmp_L0/memory_leak.py << 'EOF'
import torch

print("Allocating GPU memory without freeing...")
leaked_tensors = []

# Allocate one very large tensor
x = torch.randn(50000, 50000, device='cuda')

# Compute and display its memory footprint
num_elements = x.numel()  # total number of elements
bytes_per_element = x.element_size()  # bytes per element (e.g., 4 for float32)
total_bytes = num_elements * bytes_per_element
total_mb = total_bytes / 1024**2

print(f"Tensor shape: {tuple(x.shape)}")
print(f"Data type: {x.dtype}")
print(f"Elements: {num_elements:,}")
print(f"Bytes per element: {bytes_per_element}")
print(f"Total memory: {total_mb:.2f} MB")

# Allocate more tensors to simulate a leak
for i in range(5):
    x = torch.randn(20000, 20000, device='cuda')
    leaked_tensors.append(x)
    print(f"Allocated tensor {i+1}: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

print("\nScript ended, but memory still allocated!")
print("Check with: nvidia-smi")
input("Press Enter to exit and release memory...")
EOF


echo "Run: python ./tmp_L0/memory_leak.py"
echo "Then: nvidia-smi"
echo ""
read -p "Press Enter to continue..."
echo ""

# Exercise 5
echo "Exercise 5: Optimizing GPU Utilization in 3D Training"
echo "======================================================"
echo ""

# --- Introductory explanation ---
echo "📘 In this exercise, you’ll learn how to *optimize* GPU memory utilization"
echo "and training speed using MONAI and PyTorch options such as:"
echo "   • Mixed Precision Training (AMP)"
echo "   • Batch size and input resolution scaling"
echo "   • Optimizer choice (Adam vs SGD)"
echo ""
echo "These adjustments affect how efficiently GPU resources are used."
echo "--------------------------------------------------------------"
echo ""

# --- Python training script ---
cat > ./tmp_L0/mock_training_opt.py << 'EOF'
import torch
import time
import argparse
from monai.networks.nets import BasicUNet
from monai.losses import DiceLoss
from monai.data import MetaTensor

# -------------------------------
# Argument parser for optimization control
# -------------------------------
parser = argparse.ArgumentParser(description="Optimize GPU usage in 3D MONAI training.")
parser.add_argument("--batch_size", type=int, default=2, help="Mini-batch size.")
parser.add_argument("--vol_size", type=int, default=96, help="Cubic input size (voxels per side).")
parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision (AMP).")
parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer type.")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Model setup
# -------------------------------
model = BasicUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    features=(16, 32, 64, 128, 256, 32),
).to(device)

criterion = DiceLoss(sigmoid=True)
optimizer = (
    torch.optim.Adam(model.parameters(), lr=1e-4)
    if args.optimizer == "adam"
    else torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
)

scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

print("Training MONAI BasicUNet with GPU optimization flags...")
print(f"  ➤ Batch size: {args.batch_size}")
print(f"  ➤ Volume size: {args.vol_size}³")
print(f"  ➤ Optimizer: {args.optimizer}")
print(f"  ➤ AMP enabled: {args.amp}")
print(f"  ➤ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print("--------------------------------------------------------------")

# -------------------------------
# Mock training loop
# -------------------------------
for epoch in range(2):
    print(f"\nEpoch {epoch+1}/2")
    for batch in range(5):
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
            mem_res = torch.cuda.memory_reserved() / 1024**2
            print(f"  Batch {batch}/5 | Loss: {loss.item():.4f} | Alloc: {mem_alloc:.0f} MB | Reserved: {mem_res:.0f} MB")

        time.sleep(0.3)

print("\n✅ Optimized training complete!")
EOF

# --- Educational optimization hints ---
echo "💡 HINTS — GPU Optimization Strategies"
echo ""
echo "1️⃣ **Batch size scaling**:"
echo "   - Increasing batch size raises GPU memory use and can improve utilization."
echo "   Example:"
echo "     python ./tmp_L0/mock_training_opt.py --batch_size 4"
echo ""
echo "2️⃣ **Input resolution**:"
echo "   - Larger input volumes (e.g., 128³) increase computation and memory."
echo "   Example:"
echo "     python ./tmp_L0/mock_training_opt.py --vol_size 128"
echo ""
echo "3️⃣ **Mixed Precision (AMP)**:"
echo "   - Use float16 where possible to save memory and boost throughput."
echo "   Example:"
echo "     python ./tmp_L0/mock_training_opt.py --amp"
echo ""
echo "4️⃣ **Optimizer choice:**"
echo "   - Adam uses more memory (stores moments) but converges faster."
echo "   - SGD uses less memory but may need more epochs."
echo "   Example:"
echo "     python ./tmp_L0/mock_training_opt.py --optimizer sgd"
echo ""
echo "5️⃣ **Combine optimizations:**"
echo "   Example for full optimization:"
echo "     python ./tmp_L0/mock_training_opt.py --batch_size 4 --vol_size 128 --amp --optimizer sgd"
echo ""
echo "--------------------------------------------------------------"
echo "📈 Observe in another terminal:"
echo "   watch -n 1 nvidia-smi"
echo "   or: nvtop (module load nvtop/3.2.0-GCCcore-13.3.0)"
echo ""
echo "   Compare how AMP and larger batches affect GPU memory utilization and speed."
echo "--------------------------------------------------------------"
echo ""
read -p "Press Enter to continue..."
echo ""

# Exercise 6 (Updated)
echo "Exercise 6: Optimizing and Benchmarking GPU Throughput"
echo "======================================================"
echo ""


# --- Explanation ---
echo "📘 This experiment benchmarks how batch size and input volume size affect:"
echo "   • GPU memory usage (MB)"
echo "   • Time to complete one epoch (12 samples)"
echo "   • Iterations per second and samples per second"
echo ""
echo "It ensures proper GPU memory cleanup and synchronization between runs."
echo "--------------------------------------------------------------"
echo ""

# --- Python benchmark script ---
cat > ./tmp_L0/mock_training_scaling.py << 'EOF'
import torch
import time
import gc
import itertools
import matplotlib.pyplot as plt
from monai.networks.nets import BasicUNet
from monai.losses import DiceLoss
import sys
sys.path.append("./tmp_L0")
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_sizes = [1, 2, 4, 8]
vol_sizes = [64, 96, 128, 168, 192]
total_samples_per_epoch = 12
results = []

print("Benchmarking GPU scaling with MONAI BasicUNet...\n")

for bs, v in itertools.product(batch_sizes, vol_sizes):
    try:
        # Clean up any lingering allocations
        torch.cuda.empty_cache()

        model = BasicUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            features=(16, 32, 64, 128, 256, 32),
        ).to(device)
        criterion = DiceLoss(sigmoid=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        steps_per_epoch = total_samples_per_epoch // bs
        x = torch.randn(bs, 1, v, v, v, device=device)
        y = torch.randint(0, 2, (bs, 1, v, v, v), device=device, dtype=torch.float32)

        # Warmup iteration (stabilize memory and cuDNN autotuning)
        with torch.no_grad():
            _ = model(x)

        torch.cuda.synchronize()
        start_time = time.time()

        # One simulated epoch (covering ~12 samples total)
        for _ in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                out = model(x)
                loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

        elapsed_epoch = time.time() - start_time
        iter_time = elapsed_epoch / steps_per_epoch
        it_per_sec = 1.0 / iter_time
        samples_per_sec = (bs * steps_per_epoch) / elapsed_epoch
        mem = torch.cuda.max_memory_allocated() / 1024**2

        results.append((bs, v, mem, elapsed_epoch, it_per_sec, samples_per_sec))
        print(f"Batch={bs}, Vol={v}³ -> {mem:.0f} MB | "
              f"Epoch={elapsed_epoch:.2f}s | {it_per_sec:.2f} it/s | {samples_per_sec:.2f} samples/s")

        # Cleanup
        del model, x, y, out, loss
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    except RuntimeError as e:
        print(f"❌ OOM for Batch={bs}, Vol={v}³")
        results.append((bs, v, None, None, None, None))
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# --- Plot results ---
valid = [(b, v, m, e, it, sp) for b, v, m, e, it, sp in results if m is not None]
if not valid:
    print("No valid results to plot.")
    exit()

bs_vals, vol_vals, mem_vals, epoch_times, iters_sec, samples_sec = zip(*valid)

# Plot 1: GPU Memory vs Volume
plt.figure(figsize=(8,6))
for bs in sorted(set(bs_vals)):
    xs = [v for (b,v,_,_,_,_) in valid if b==bs]
    ys = [m for (b,v,m,_,_,_) in valid if b==bs]
    plt.plot(xs, ys, '-o', label=f'Batch {bs}')
plt.title("GPU Memory vs Volume Size")
plt.xlabel("Volume Size (voxels per side)")
plt.ylabel("Max GPU Memory (MB)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./chapter0_track_gpus/figures/memory_scaling.png")
plt.close()

# Plot 2: Time per Epoch
plt.figure(figsize=(8,6))
for bs in sorted(set(bs_vals)):
    xs = [v for (b,v,_,_,_,_) in valid if b==bs]
    ys = [e for (b,v,_,e,_,_) in valid if b==bs]
    plt.plot(xs, ys, '-o', label=f'Batch {bs}')
plt.title("Epoch Time (12 samples) vs Volume Size")
plt.xlabel("Volume Size (voxels per side)")
plt.ylabel("Seconds per Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./chapter0_track_gpus/figures/epoch_time_scaling.png")
plt.close()

# Plot 3: Throughput (Samples/s)
plt.figure(figsize=(8,6))
for bs in sorted(set(bs_vals)):
    xs = [v for (b,v,_,_,_,_) in valid if b==bs]
    ys = [sp for (b,v,_,_,_,sp) in valid if b==bs]
    plt.plot(xs, ys, '-o', label=f'Batch {bs}')
plt.title("Throughput (Samples per Second)")
plt.xlabel("Volume Size (voxels per side)")
plt.ylabel("Samples per Second")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./chapter0_track_gpus/figures/throughput_scaling.png")
plt.close()

print("\n✅ Benchmark complete!")
print("Plots saved to:")
print("  ./tmp_L0/memory_scaling.png")
print("  ./tmp_L0/epoch_time_scaling.png")
print("  ./tmp_L0/throughput_scaling.png")
EOF

# --- Hints for interpretation ---
echo "💡 HINTS — How to Analyze the Results"
echo ""
echo "1️⃣ The *Memory plot* shows how GPU RAM grows ~linearly with batch size"
echo "   and roughly ~cubicly with input volume dimension."
echo ""
echo "2️⃣ The *Epoch time plot* shows how long it takes to process 12 samples."
echo "   You can see how doubling batch size reduces the number of steps per epoch."
echo ""
echo "3️⃣ The *Throughput plot* (samples/s) shows efficiency: higher is better."
echo "   The optimal configuration is the one with the highest throughput"
echo "   before hitting OOM or excessive memory usage."
echo ""
echo "Run the benchmark:"
echo "   python ./tmp_L0/mock_training_scaling.py"
echo ""
echo "Then open the plots:"
echo "   eog ./tmp_L0/memory_scaling.png &"
echo "   eog ./tmp_L0/epoch_time_scaling.png &"
echo "   eog ./tmp_L0/throughput_scaling.png &"
echo "--------------------------------------------------------------"
echo ""
read -p "Press Enter to continue..."
echo ""

# Exercise 7
echo "Exercise 7: Precision and Optimization Strategies in 3D Training"
echo "================================================================="
echo ""

# --- Introductory explanation ---
echo "📘 This exercise compares precision and optimization strategies for GPU training."
echo ""
echo "You will benchmark the same 3D model using different numerical precisions:"
echo "   • Float32 (FP32) — baseline"
echo "   • Mixed Precision (AMP) — dynamic casting for speed"
echo "   • BFloat16 (BF16) — stable reduced precision on newer GPUs"
echo "   • Float16 (FP16) — highest compression, fastest but less stable"
echo ""
echo "You will measure:"
echo "   • GPU memory (MB)"
echo "   • Time per iteration (s/it)"
echo "   • Throughput (samples/s)"
echo ""
echo "--------------------------------------------------------------"
echo ""

# --- Python script ---

cat > ./tmp_L0/mock_precision_benchmark.py << 'EOF'
import torch
import time
import matplotlib.pyplot as plt
from monai.networks.nets import BasicUNet
from monai.losses import DiceLoss

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check GPU precision support
gpu_name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
supports_bf16 = capability >= (8, 0)  # Ampere or newer
supports_fp16 = capability >= (7, 0)

print(f"Detected GPU: {gpu_name} (Compute {capability[0]}.{capability[1]})")
print(f"Supports BF16: {supports_bf16}, Supports FP16: {supports_fp16}\n")

precisions = ["fp32", "amp", "bf16", "fp16"]
batch_size = 2
vol_size = 96
steps = 8

results = []

for prec in precisions:
    try:
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

        # --- Configure precision mode ---
        if prec == "fp32":
            use_amp = False
            use_scaler = False
            dtype = torch.float32
        elif prec == "amp":
            use_amp = True
            use_scaler = True
            dtype = torch.float32
        elif prec == "bf16":
            use_amp = True
            use_scaler = False
            dtype = torch.bfloat16
            if not supports_bf16:
                raise RuntimeError("BF16 not supported on this GPU.")
        elif prec == "fp16":
            use_amp = False
            use_scaler = False
            dtype = torch.float16
            if not supports_fp16:
                raise RuntimeError("FP16 not supported on this GPU.")
            model = model.half()
        else:
            continue

        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
        autocast_dtype = torch.bfloat16 if prec == "bf16" else torch.float16

        x = torch.randn(batch_size, 1, vol_size, vol_size, vol_size, device=device, dtype=dtype)
        y = torch.randint(0, 2, (batch_size, 1, vol_size, vol_size, vol_size),
                          device=device, dtype=dtype)

        print(f"{prec.upper():>6s} | BasicUNet features: (32, 32, 64, 128, 256, 32).")

        # Warm-up
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
        samples_per_sec = batch_size / elapsed
        results.append((prec, mem, elapsed, samples_per_sec))

        print(f"   {prec.upper():>4s} | {mem:>8.0f} MB | {elapsed:.3f} s/it | {samples_per_sec:.2f} samples/s\n")

        del model, x, y, out, loss
        torch.cuda.empty_cache()

    except RuntimeError as e:
        print(f"❌ {prec.upper()} failed: {e}")
        results.append((prec, None, None, None))
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

# --- Plot results ---
valid = [(p, m, t, s) for p, m, t, s in results if m is not None]
if not valid:
    print("No valid results to plot.")
    exit()

prec_labels, mem_vals, time_vals, samp_vals = zip(*valid)

plt.figure(figsize=(8,5))
plt.bar(prec_labels, mem_vals, color=["steelblue","orange","green","purple"])
plt.title("GPU Memory Usage by Precision")
plt.ylabel("Memory (MB)")
plt.xlabel("Precision Mode")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("./chapter0_track_gpus/figures/precision_memory.png")
plt.close()

plt.figure(figsize=(8,5))
plt.bar(prec_labels, samp_vals, color=["steelblue","orange","green","purple"])
plt.title("Throughput (Samples/s) by Precision")
plt.ylabel("Samples per Second")
plt.xlabel("Precision Mode")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("./chapter0_track_gpus/figures/precision_throughput.png")
plt.close()

print("✅ Benchmark complete!")
print("Plots saved to:")
print("   /tmp/precision_memory.png")
print("   /tmp/precision_throughput.png")

# --- Summary printout ---
best_prec = max(valid, key=lambda x: x[3])
print(f"\n🚀 Recommended Precision for {gpu_name}: {best_prec[0].upper()} "
      f"(~{best_prec[1]:.0f} MB, {best_prec[3]:.2f} samples/s)")
EOF

# --- Educational explanation ---
echo "💡 HINTS — Understanding Precision Optimization"
echo ""
echo "1️⃣ **FP32 (Single Precision)**:"
echo "   - Default for PyTorch. Highest accuracy but largest memory footprint."
echo "   - Used as baseline."
echo ""
echo "2️⃣ **AMP (Automatic Mixed Precision)**:"
echo "   - Dynamically uses FP16 where safe and FP32 for numerically sensitive ops."
echo "   - 30–50% memory reduction and ~1.5×–2× speedup on modern GPUs."
echo ""
echo "3️⃣ **BF16 (BFloat16)**:"
echo "   - Similar speed to FP16 but better numerical stability."
echo "   - Supported on A100, H100, and newer architectures."
echo ""
echo "4️⃣ **FP16 (Half Precision)**:"
echo "   - Full half precision training. Fast but prone to underflow/overflow."
echo "   - Usually only stable with loss scaling or AMP."
echo ""
echo "Run the script:"
echo "   python ./tmp_L0/mock_precision_benchmark.py"
echo ""
echo "Then open the plots:"
echo "   eog ./tmp_L0/precision_memory.png &"
echo "   eog ./tmp_L0/precision_throughput.png &"
echo ""
echo "--------------------------------------------------------------"
echo "💪 Exercise Discussion:"
echo ""
echo "• Which precision gives the best throughput/memory trade-off?"
echo "• On your GPU (e.g., A100, L40S, V100), does BF16 behave like FP16 or FP32?"
echo "• What’s the ideal precision for stability vs. efficiency?"
echo ""
read -p "Press Enter to continue..."
echo ""

# --- Additional optimization strategies ---
echo "💡 ADVANCED OPTIMIZATION STRATEGIES"
echo ""
echo "1️⃣ **Gradient Accumulation**"
echo "   - Simulate larger effective batch size without increasing memory."
echo "   Example:"
echo "     for step, batch in enumerate(loader):"
echo "         loss = model(batch) / grad_accum_steps"
echo "         loss.backward()"
echo "         if (step+1) % grad_accum_steps == 0:"
echo "             optimizer.step(); optimizer.zero_g"

# Exercise 8
echo "Exercise 8: Precision + Gradient Accumulation Optimization"
echo "==========================================================="
echo ""

# --- Introductory explanation ---
echo "📘 In this final optimization exercise, you'll explore how precision and"
echo "gradient accumulation together affect GPU memory and throughput."
echo ""
echo "By accumulating gradients over multiple micro-batches, you simulate a"
echo "larger effective batch size *without increasing memory use*."
echo ""
echo "You’ll test:"
echo "   • Precisions: FP32, AMP, BF16, FP16"
echo "   • Gradient accumulation: 1×, 2×, 4× steps"
echo ""
echo "Metrics recorded per configuration:"
echo "   - GPU Memory (MB)"
echo "   - Time per epoch (12 samples)"
echo "   - Iterations/s and Samples/s"
echo ""
echo "--------------------------------------------------------------"
echo ""
# --- Python benchmark script ---
cat > ./tmp_L0/mock_precision_accumulation.py << 'EOF'
import torch
import time
import itertools
import matplotlib.pyplot as plt
import gc
from monai.networks.nets import BasicUNet
from monai.losses import DiceLoss

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

precisions = ["fp32", "amp", "bf16"]
accum_steps_list = [1, 2, 4, 8, 16]
batch_size = 2
vol_size = 96
total_samples = 12

results = []

print("Benchmarking precision + gradient accumulation...\n")

for prec, accum_steps in itertools.product(precisions, accum_steps_list):
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = BasicUNet(spatial_dims=3, in_channels=1, out_channels=1).to(device)
        criterion = DiceLoss(sigmoid=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        autocast_dtype = (
            torch.float16 if prec == "amp"
            else torch.bfloat16 if prec == "bf16"
            else torch.float32
        )
        autocast_enabled = prec in ["amp", "bf16"]

        use_scaler = (prec == "amp")
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

        steps = total_samples // batch_size
        x = torch.randn(batch_size, 1, vol_size, vol_size, vol_size, device=device)
        y = torch.randint(0, 2, (batch_size, 1, vol_size, vol_size, vol_size),
                          device=device, dtype=torch.float32)

        torch.cuda.synchronize()
        start_time = time.time()

        for i in range(steps):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accum_steps):
                with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=autocast_dtype):
                    out = model(x)
                    loss = criterion(out, y) / accum_steps

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

        elapsed = time.time() - start_time
        mem = torch.cuda.max_memory_allocated() / 1024**2
        iters = steps
        iters_per_sec = iters / elapsed
        samples_per_sec = (batch_size * iters * accum_steps) / elapsed

        results.append((prec, accum_steps, mem, elapsed, iters_per_sec, samples_per_sec))
        print(f"{prec.upper():>5s} | accum={accum_steps:<2d} | {mem:>7.0f} MB | "
              f"{elapsed:>6.2f}s/epoch | {samples_per_sec:.2f} samples/s")

        del model, x, y, out, loss
        gc.collect()
        torch.cuda.empty_cache()

    except RuntimeError as e:
        print(f"❌ OOM for {prec.upper()} accum={accum_steps}")
        results.append((prec, accum_steps, None, None, None, None))
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

# --- Plot results ---
valid = [(p, a, m, e, it, sp) for p, a, m, e, it, sp in results if m is not None]
if not valid:
    print("No valid results to plot.")
    exit()

prec_vals, accum_vals, mem_vals, epoch_time, iter_s, samples_s = zip(*valid)

# Plot 1: Memory vs Accumulation
plt.figure(figsize=(8, 6))
for prec in sorted(set(prec_vals)):
    xs = [a for (p, a, _, _, _, _) in valid if p == prec]
    ys = [m for (p, a, m, _, _, _) in valid if p == prec]
    plt.plot(xs, ys, '-o', label=f'{prec.upper()}')
plt.title("GPU Memory vs Gradient Accumulation")
plt.xlabel("Gradient Accumulation Steps")
plt.ylabel("Max GPU Memory (MB)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./chapter0_track_gpus/figures/memory_accumulation.png")
plt.close()

# Plot 2: Throughput (Samples/s)
plt.figure(figsize=(8, 6))
for prec in sorted(set(prec_vals)):
    xs = [a for (p, a, _, _, _, _) in valid if p == prec]
    ys = [s for (p, a, _, _, _, s) in valid if p == prec]
    plt.plot(xs, ys, '-o', label=f'{prec.upper()}')
plt.title("Throughput (Samples/s) vs Gradient Accumulation")
plt.xlabel("Gradient Accumulation Steps")
plt.ylabel("Samples per Second")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./chapter0_track_gpus/figures/throughput_accumulation.png")
plt.close()

print("\n✅ Benchmark complete!")
print("Plots saved to:")
print("  ./tmp_L0/memory_accumulation.png")
print("  ./tmp_L0/throughput_accumulation.png")
EOF

# --- Echo explanation and hints ---
echo "💡 HINTS — Gradient Accumulation and Precision Trade-offs"
echo ""
echo "1️⃣ **Gradient Accumulation**:"
echo "   - Splits a large batch into smaller micro-batches processed sequentially."
echo "   - Accumulates gradients before the optimizer update."
echo "   - Memory stays constant, but total epoch time increases slightly."
echo ""
echo "2️⃣ **AMP (Mixed Precision)**:"
echo "   - Further reduces memory and speeds up forward/backward passes."
echo "   - Works well together with accumulation to simulate huge batches."
echo ""
echo "3️⃣ **BF16 Precision**:"
echo "   - Offers a stable middle ground: almost FP32 accuracy with FP16 memory usage."
echo ""
echo "4️⃣ **Expected Results:**"
echo "   - Memory stays roughly constant as accumulation increases."
echo "   - Throughput (samples/s) initially rises then plateaus."
echo "   - Larger accumulation mimics a bigger batch with no extra GPU cost."
echo ""
echo "Run the exercise:"
echo "   python ./tmp_L0/mock_precision_accumulation.py"
echo ""
echo "Then open plots:"
echo "   eog ./tmp_L0/memory_accumulation.png &"
echo "   eog ./tmp_L0/throughput_accumulation.png &"
echo ""
echo "--------------------------------------------------------------"
echo "💪 Discussion:"
echo ""
echo "• Which precision yields the best throughput-memory balance?"
echo "• How does accumulation affect total epoch time?"
echo "• Can AMP + accumulation allow you to train larger 3D models on smaller GPUs?"
echo ""
read -p "Press Enter to continue..."
echo ""

# --- Summary ---
echo "=========================================="
echo "Exercise Summary"
echo "=========================================="
echo ""
echo "You've learned to:"
echo "  ✓ Combine precision control and gradient accumulation"
echo "  ✓ Benchmark GPU memory vs throughput in realistic 3D workloads"
echo "  ✓ Simulate large batch sizes on limited GPUs"
echo ""
echo "Key Takeaways:"
echo "  • Gradient accumulation increases effective batch size without memory growth"
echo "  • AMP and BF16 reduce memory and improve speed"
echo "  • Best efficiency = AMP + accumulation + mixed optimizer (AdamW/SGD)"
echo ""
echo "Next: Chapter 4 – Distributed Training and Model Sharding"
echo ""
echo "Clean up:"
echo "rm ./tmp_L0/mock_precision_accumulation.py ./tmp_L0/memory_accumulation.png ./tmp_L0/throughput_accumulation.png"
echo ""
