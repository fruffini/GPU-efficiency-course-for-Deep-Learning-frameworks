#!/bin/bash

echo "=========================================="
echo "Chapter 0: SLURM GPU Information"
echo "=========================================="
echo ""

echo "1. GPU Nodes Overview"
echo "====================="
sinfo -o "%20N %10c %10m %25f %10G" | grep gpu
echo ""

echo "2. Available GPUs by Partition"
echo "==============================="
sinfo -p gpu -O "partition,nodes,cpus,memory,gres,gresused,available"
echo ""

echo "3. Detailed GPU Node Information"
echo "================================="
scontrol show nodes | grep -E "NodeName|Gres|CPUTot|RealMemory|State"
echo ""

echo "4. Current GPU Job Queue"
echo "========================"
squeue -p gpu -o "%.18i %.9P %.20j %.8u %.8T %.10M %.6D %.4C %b"
echo ""

echo "5. Your Running GPU Jobs"
echo "========================"
squeue -u $USER -o "%.18i %.9P %.20j %.8T %.10M %.6D %.4C %b" | grep gpu
echo ""

echo "6. GPU Utilization Summary"
echo "=========================="
sinfo -p gpu -h -o "%D %G" | awk '{
    split($2, a, ":");
    if (length(a) > 1) {
        split(a[2], b, "(");
        if (length(b) > 1) {
            split(b[2], c, ")");
            total += b[1]
            used += c[1]
        }
    }
}
END {
    if (total > 0) {
        printf "Used: %d / %d GPUs (%.1f%%)\n", used, total, (used/total)*100
    }
}'

echo ""

echo "=========================================="
echo "Useful SLURM Commands"
echo "=========================================="
echo ""
echo "# Check A40 (or any) availability"
echo "sinfo -p gpu -o '%50N %10G' | grep 'a40'"
echo ""
echo "# Request specific GPU type"
echo "sbatch --gres=gpu:a100:2 --partition=gpu job.sh"
echo ""
echo "# Check job pending reason"
echo "squeue -j <JOB_ID> --start"
echo "squeue -u ruffini"



echo "# Check job pending reason"
echo "squeue -j <JOB_ID> --start"





echo ""
echo "# Monitor running job GPU"
echo "srun --jobid=<JOB_ID> --pty nvidia-smi"
echo ""
echo "# Cancel GPU jobs"
echo "scancel -u $USER --partition=gpu"
echo ""
