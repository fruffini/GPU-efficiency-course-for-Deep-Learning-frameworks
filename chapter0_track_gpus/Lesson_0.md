# Chapter 0: Track Your GPUs

## Learning Objectives

By the end of this chapter, you will:
- Understand GPU hardware specifications and capabilities
- Monitor real-time GPU utilization (memory, compute, temperature)
- Use `nvidia-smi` effectively for debugging
- Query SLURM for GPU allocation and availability
- Identify common GPU bottlenecks in medical imaging workflows

## Prerequisites

- Access to a machine with NVIDIA GPU(s)
- NVIDIA drivers installed
- SLURM cluster access (for exercises 03-04)
- Python 3.8+ with `pynvml` library

## Quick Start
```bash
# Activate your virtual environment
source /mimer/NOBACKUP/groups/naiss2023-6-336/AIDA_multimodal_F&C/merlin_env


# Run basic GPU check
bash 01_nvidia_smi_nvtop_basics.sh

# Monitor GPU in real-time
python 02_gpu_monitoring.py

# Check SLURM GPU availability (if on cluster)
bash 03_slurm_gpu_info.sh

# Complete practical exercises
bash 04_practical_exercises.sh
```

## Why GPU Monitoring Matters in Medical Imaging

Medical imaging workloads have unique characteristics:
- **Large memory footprint**: A single 3D CT scan can be 512×512×300 voxels (300MB+ in float32)
- **Irregular batch sizes**: Memory constraints often force small batch sizes
- **CPU bottlenecks**: I/O-bound data loading can starve the GPU
- **Unexpected OOM**: Out-of-memory errors waste hours of training time

**Real scenario**: You submit a 12-hour training job with 8 GPUs. After 10 hours, you discover only 1 GPU was utilized at 15% while others were idle. Understanding GPU monitoring prevents this waste.

## Key Metrics to Track

| Metric | Ideal Range | Warning Signs |
|--------|-------------|---------------|
| **GPU Utilization** | >80% | <50% = CPU bottleneck or small batches |
| **Memory Usage** | 70-90% | >95% = imminent OOM; <40% = underutilized |
| **Temperature** | <80°C | >85°C = throttling risk |
| **Power Draw** | Near TDP | Much lower = underutilized |
| **SM Clock** | Near max boost | Reduced = thermal/power throttling |

## Files in This Chapter

1. **01_nvidia_smi_basics.sh**: Essential `nvidia-smi` commands
2. **02_gpu_monitoring.py**: Real-time monitoring with Python
3. **03_slurm_gpu_info.sh**: Query SLURM for GPU resources
4. **04_practical_exercises.sh**: Hands-on exercises with solutions