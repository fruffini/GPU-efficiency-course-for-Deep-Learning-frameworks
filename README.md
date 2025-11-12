# GPU Efficiency in SLURM: A Practical Course for Medical Imaging

## Directory Structure

```

gpu_slurm_medical_imaging/
├── chapter0_track_gpus/
│   ├── README.md
│   ├── 01_nvidia_smi_basics.sh
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
# create venv
python3 -m venv .venv
# activate
source .venv/bin/activate
# upgrade pip and install requirements if present
pip install --upgrade pip
```

2) Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

3) GPU-aware libraries
- For PyTorch with CUDA, prefer the official selector at https://pytorch.org to get the correct install command, for example:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
- Ensure CUDA drivers on the host match the chosen CUDA runtime. If using SLURM on a cluster, check available CUDA module versions (e.g., module avail cuda) or ask sysadmin.

4) Optional helper script
- A convenience script is provided at utils/setup_venv.sh to automate creation and install requirements. Run: bash utils/setup_venv.sh

## Medical Imaging Focus

- **Data Format**: NIfTI, DICOM (3D volumes, not 2D images)
- **Task Examples**: Segmentation (3D U-Net), Detection (YOLO on CT), Classification (Survival prediction)
- **Memory Challenges**: 3D volumes can be 200MB+ per sample
- **Practical Scenario**: Loading 10K chest CTs for training on 8GB GPU memory
