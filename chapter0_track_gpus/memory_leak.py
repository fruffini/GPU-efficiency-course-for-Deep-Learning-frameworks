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
