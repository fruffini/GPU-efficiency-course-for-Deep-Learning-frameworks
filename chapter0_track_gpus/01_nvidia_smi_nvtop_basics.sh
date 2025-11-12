#!/bin/bash
# Chapter 0: nvidia-smi Basics
# This script demonstrates essential nvidia-smi commands for GPU monitoring

echo "1. Basic GPU Information"
echo "========================"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv


echo "2. Real-time GPU Utilization (refresh every 1 second for 5 iterations)"
echo "======================================================================"
echo "Press Ctrl+C to stop if needed..."
nvidia-smi dmon -c 5 -s mu


echo "3. Memory Usage Details"
echo "======================="
nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv


echo "4. GPU Utilization and Temperature"
echo "==================================="
nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv


echo "5. Process Information (what's running on GPUs)"
echo "==============================================="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv


echo "6. Detailed View for Debugging"
echo "==============================="
nvidia-smi

echo "# Watch GPU usage in real-time (updates every 2 seconds)"
watch -n 2 nvidia-smi


echo "# Check if specific GPU (e.g., GPU 0) is available"
nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits

# Additional commands
echo "# Find which process is using GPU memory"
nvidia-smi pmon -c 1


echo "=========================================="
echo "NVTOP: An Interactive GPU Monitoring Tool"
echo "=========================================="

echo "Launch NVTOP:"
module load nvtop/3.2.0-GCCcore-13.3.0
nvtop


echo "=========================================="
echo "Environment Variables for GPU Selection"
echo "=========================================="


echo "# Use only GPU 0"
export CUDA_VISIBLE_DEVICES=0

echo "# Use GPUs 0 and 2"
export CUDA_VISIBLE_DEVICES=0,2

echo "# Hide all GPUs (for CPU-only testing)"
export CUDA_VISIBLE_DEVICES=''

