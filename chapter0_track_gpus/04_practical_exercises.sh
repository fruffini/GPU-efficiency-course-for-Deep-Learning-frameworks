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

cat > ./chapter0_track_gpus/gpu_load.py << 'EOF'
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
cat > ./chapter0_track_gpus/memory_leak.py << 'EOF'
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
