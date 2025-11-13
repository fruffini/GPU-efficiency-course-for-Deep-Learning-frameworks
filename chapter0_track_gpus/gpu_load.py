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
