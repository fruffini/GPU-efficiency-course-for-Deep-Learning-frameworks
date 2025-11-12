# GPU Efficiency in SLURM: A Practical Course for Medical Imaging

## Directory Structure

```

gpu_slurm_medical_imaging/
├── chapter0_track_gpus/
│   ├── README.md
│   ├── 01_nvidia_smi_nvtop_basics.sh
│   ├── 02_gpu_monitoring.py
│   ├── 03_slurm_gpu_info.sh
│   └── 04_practical_exercises.sh
│
├── chapter1_data_loading/
│   ├── README.md
│   ├── 01_naive_loading.py
│   ├── 02_efficient_loading_medical.py
│   ├── 03_prefetching_pipeline.py
│   ├── 04_benchmark_comparison.py
│   └── data/
│       └── sample_ct_volume.nii.gz (placeholder)
│
├── chapter2_batch_processing/
│   ├── README.md
│   ├── 01_memory_profiling.py
│   ├── 02_batch_accumulation.py
│   ├── 03_adaptive_batch_sizing.py
│   └── submit_batch_job.sh
│
├── chapter3_distributed_training/
│   ├── README.md
│   ├── 01_ddp_medical_imaging.py
│   ├── 02_multi_gpu_inference.py
│   └── submit_multi_gpu.sh
│
├── chapter4_profiling_optimization/
│   ├── README.md
│   ├── 01_profiling_bottlenecks.py
│   ├── 02_mixed_precision_training.py
│   ├── 03_gradient_checkpointing.py
│   └── analyze_profile.py
│
├── chapter5_slurm_best_practices/
│   ├── README.md
│   ├── 01_basic_gpu_job.sh
│   ├── 02_optimized_job.sh
│   ├── 03_gpu_and_cpu_binding.sh
│   └── 04_job_monitoring.py
│
└── utils/
    ├── gpu_utils.py
    ├── medical_data_utils.py
    └── slurm_helper.sh
```

## Course Outline

| Chapter | Topic | Learning Goals |
|---------|-------|-----------------|
| **0** | Track your GPUs | Understand GPU hardware, monitor utilization, use nvidia-smi, query SLURM |
| **1** | Data Loading Strategies | Load medical imaging efficiently, implement prefetching, avoid GPU idle time |
| **2** | Batch Processing & Memory | Profile memory usage, adaptive batching, reduce out-of-memory errors |
| **3** | Distributed Training | Multi-GPU training, DDP setup, scaling across nodes |
| **4** | Profiling & Optimization | Find bottlenecks, mixed precision, reduce training time by 30-50% |
| **5** | SLURM Best Practices | Write efficient job scripts, resource allocation, job monitoring |

## Setup: Virtual environment

Add a reproducible Python virtual environment before running examples and scripts. Below are short instructions and an optional helper script.

1) Linux / macOS (bash)
```bash
cd /mimer/NOBACKUP/groups/naiss2023-6-336/GPU_efficiency_course
# ALVIS python module search
module load Python/3.12.3-GCCcore-13.3.0
# We also import the nvtop module for later
module load nvtop/3.2.0-GCCcore-13.3.0
# And also load virtualenv
module load virtualenv/20.26.2-GCCcore-13.3.0

```
2) Create and activate virtual environment

# create venv
```bash
virtualenv --system-site-packages .venv
# activate (im using an old env )
source "/mimer/NOBACKUP/groups/naiss2023-6-336/AIDA_multimodal_F&C/merlin_env/bin/activate"
# upgrade pip and install requirements if present (1 time command)
pip install --upgrade pip'''
```
3) If exists, just load the modules and activate the venv

```bash
cd /mimer/NOBACKUP/groups/naiss2023-6-336/GPU_efficiency_course
module purge
module load Python/3.12.3-GCCcore-13.3.0
module load nvtop/3.2.0-GCCcore-13.3.0
module load virtualenv/20.26.2-GCCcore-13.3.0
# activate
source "/mimer/NOBACKUP/groups/naiss2023-6-336/AIDA_multimodal_F&C/merlin_env/bin/activate"
source .venv/bin/activate
# Install required packages
pip install pynvml psutil matplotlib torch monai
```



3) LAUNCH JOB and CONNECT
- Use `srun` or `sbatch` to launch your scripts on SLURM with GPU requests, e.g.:

- "-N": number of nodes
- "-p": partition name (alvis)
- "--cpus-per-task=16": number of CPU cores per task (important for data loading)
- "--ntasks-per-node=": number of tasks per node (usually 1 task / GPU)
- "-t": time limit
- "--gpus-per-node": number and type of GPUs per node
- "--pty bash": interactive bash session
- ```bash
  srun -A NAISS2023-5-577 -p alvis -N 1 -t 02:30:00 --cpus-per-task=16 --gpus-per-node=A40:1 --ntasks-per-node=1 --pty bash # This command requests 1 A40 GPU for an interactive bash session (it launches a terminal) -- Be careful about usage, this job can be killed for inefficiency by NAISS!
  ```
# For srun, the job "lives" in the termoinal -pty bash opened when you launch the command, if you dusconnect from it, the job is directly canceled!

4) GPU-aware libraries
- For PyTorch with CUDA, prefer the official selector at https://pytorch.org to get the correct install command, for example:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
- Ensure CUDA drivers on the host match the chosen CUDA runtime. If using SLURM on a cluster, check available CUDA module versions (e.g., module avail cuda) or ask sysadmin.
