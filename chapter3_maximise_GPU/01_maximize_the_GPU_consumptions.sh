#!/bin/bash
# ================================================================
# LESSON — MAXIMIZE GPU CONSUMPTION & EFFICIENCY
# Complete Guide to GPU Resource Optimization
# ================================================================

set -e

# ---------- COLORS ----------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; MAGENTA='\033[0;35m'; CYAN='\033[0;36m'; NC='\033[0m'

print_header() { echo -e "\n${CYAN}================================================================${NC}\n${CYAN}$1${NC}\n${CYAN}================================================================${NC}\n"; }
print_section() { echo -e "\n${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n${MAGENTA}$1${NC}\n${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"; }
print_info() { echo -e "${BLUE}ℹ${NC} $1"; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_command() { echo -e "${GREEN}$${NC} ${CYAN}$1${NC}"; }
pause() { echo ""; read -p "Press ENTER to continue..."; echo ""; }

# ================================================================
# LESSON INTRODUCTION
# ================================================================
clear
print_header "Lesson: Maximizing GPU Consumption & Efficiency"

cat << 'EOF'
    __  __            _           _         ____  ____  _   _
   |  \/  | __ ___  _(_)_ __ ___ (_)_______| ___||  _ \| | | |
   | |\/| |/ _` \ \/ / | '_ ` _ \| |_  / _ \___ \| |_) | | | |
   | |  | | (_| |>  <| | | | | | | |/ /  __/___) |  __/| |_| |
   |_|  |_|\__,_/_/\_\_|_| |_| |_|_/___\___|____/|_|   \___/

🎯 OBJECTIVE:
  Master GPU resource optimization through:
    1. Understanding GPU utilization bottlenecks
    2. Parallel training strategies
    3. Memory-efficient caching
    4. Real-time monitoring and adjustment
    5. Multi-process orchestration

📊 COMMON GPU UTILIZATION SCENARIOS:

  Scenario A: Low Utilization (20-40%)
  ┌────────────────────────────────────────┐
  │ GPU: ████░░░░░░░░░░░░░░░░░░  20%      │
  │ CPU: ████████████░░░░░░░░░░  50%      │
  │ Problem: CPU-bound data loading        │
  │ Solution: Increase num_workers         │
  └────────────────────────────────────────┘

  Scenario B: Moderate Utilization (50-70%)
  ┌────────────────────────────────────────┐
  │ GPU: ██████████████░░░░░░░░  60%      │
  │ Mem: ████████░░░░░░░░░░░░░░  40%      │
  │ Problem: Small batch size              │
  │ Solution: Increase batch or run 2×     │
  └────────────────────────────────────────┘

  Scenario C: Optimal Utilization (80-95%)
  ┌────────────────────────────────────────┐
  │ GPU: ████████████████████░░  90%      │
  │ Mem: ██████████████████░░░░  85%      │
  │ Status: ✅ Well optimized!             │
  └────────────────────────────────────────┘

WHY PARALLEL TRAINING WORKS:
  • Medical imaging often has small batch sizes (memory limited)
  • Single process may not saturate GPU compute
  • Multiple processes share GPU, fill idle capacity
  • SmartCacheDataset keeps data in RAM (fast access)

⚠️ CRITICAL WARNINGS:
  ✗ Too many processes → CPU thrashing
  ✗ Exceeding GPU memory → OOM errors
  ✗ No monitoring → waste resources
  ✓ Always monitor: nvidia-smi + htop
EOF

pause

# ================================================================
# SYSTEM ANALYSIS
# ================================================================

print_header "Step 0: Analyze Your System"

print_section "Detecting Hardware Capabilities"

cat > chapter3_maximise_GPU/analyze_system.py << 'PYEOF'
#!/usr/bin/env python3
"""Analyze system capabilities for parallel training"""

import torch
import psutil
import os


def analyze_system():
    print("="*70)
    print("SYSTEM ANALYSIS")
    print("="*70)

    # GPU Info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_count = torch.cuda.device_count()

        print(f"\n🎮 GPU Information:")
        print(f"   Device: {gpu_name}")
        print(f"   Count: {gpu_count}")
        print(f"   Memory: {gpu_mem_total:.1f} GB")

        # Estimate parallel capacity
        estimated_per_process = 4.0  # GB per training process
        max_parallel_gpu = int(gpu_mem_total * 0.85 / estimated_per_process)

        print(f"\n   💡 Estimated capacity:")
        print(f"      {max_parallel_gpu} parallel trainings")
        print(f"      (assuming ~{estimated_per_process:.1f} GB per process)")
    else:
        print("\n❌ No GPU detected!")
        return

    # CPU Info
    cpu_physical = psutil.cpu_count(logical=False)
    cpu_logical = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()

    print(f"\n🧠 CPU Information:")
    print(f"   Physical cores: {cpu_physical}")
    print(f"   Logical cores: {cpu_logical}")
    print(f"   Frequency: {cpu_freq.current:.0f} MHz")

    # Calculate safe parallel count
    # Rule: Each process needs ~4 workers (DataLoader + SmartCache)
    workers_per_process = 4
    max_parallel_cpu = cpu_logical // workers_per_process

    print(f"\n   💡 CPU-safe parallel count:")
    print(f"      {max_parallel_cpu} parallel trainings")
    print(f"      (each using ~{workers_per_process} threads)")

    # RAM Info
    ram = psutil.virtual_memory()
    ram_total_gb = ram.total / 1024**3
    ram_available_gb = ram.available / 1024**3

    print(f"\n💾 RAM Information:")
    print(f"   Total: {ram_total_gb:.1f} GB")
    print(f"   Available: {ram_available_gb:.1f} GB")
    print(f"   Used: {ram.percent}%")

    # Estimate cache capacity
    cache_per_process = 2.0  # GB for SmartCache
    max_parallel_ram = int(ram_available_gb * 0.7 / cache_per_process)

    print(f"\n   💡 RAM-safe parallel count:")
    print(f"      {max_parallel_ram} parallel trainings")
    print(f"      (assuming ~{cache_per_process:.1f} GB cache per process)")

    # Final recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    max_safe = min(max_parallel_gpu, max_parallel_cpu, max_parallel_ram)

    print(f"""
🎯 OPTIMAL PARALLEL COUNT: {max_safe} processes

Limiting factors:
  • GPU memory: up to {max_parallel_gpu} processes
  • CPU cores: up to {max_parallel_cpu} processes
  • RAM: up to {max_parallel_ram} processes

  → Safe maximum: {max_safe} (most restrictive)

COMMANDS TO USE:
  # Launch {max_safe} parallel trainings
  for i in {{1..{max_safe}}}; do
      python3 training_script.py &
  done

  # Monitor all processes
  watch -n 1 nvidia-smi
  htop

⚠️  START WITH 2-3 PROCESSES FIRST!
   Then increase if GPU utilization is still low.
    """)

    print("="*70)


if __name__ == "__main__":
    analyze_system()
PYEOF

chmod +x chapter3_maximise_GPU/analyze_system.py
print_success "Created: analyze_system.py"
echo ""

print_info "Running system analysis..."
print_command "python3 chapter3_maximise_GPU/analyze_system.py"
echo ""

python3 chapter3_maximise_GPU/analyze_system.py

pause

# ================================================================
# STEP 1: CREATE TRAINING SCRIPTS
# ================================================================

print_header "Step 1: Create Training Scripts"

print_section "1.1: Basic Training Script"

mkdir -p chapter3_maximise_GPU

cat > chapter3_maximise_GPU/train_single.py << 'PYEOF'
#!/usr/bin/env python3
"""
Single Training Process
Basic training with monitoring
"""

import torch
import psutil
import time
import os
from pathlib import Path
from monai.data import SmartCacheDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Resized,
    ScaleIntensityRanged, RandRotate90d, ToTensord
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss


def get_transforms():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(96, 96, 96), mode="trilinear"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-200, a_max=200,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(0, 2)),
        ToTensord(keys=["image"]),
    ])


def train_single_process(
    data_dir="chapter2_data_efficiency/data/nifti_samples",
    epochs=3,
    batch_size=2,
    process_id=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pid = process_id if process_id else os.getpid()
    print(f"\n{'='*70}")
    print(f"Training Process [PID: {pid}]")
    print(f"{'='*70}")

    # Load data
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("   Run: python3 chapter2_data_efficiency/generate_sample_data.py")
        return

    imgs = sorted(data_path.glob("*.nii.gz"))[:40]
    if len(imgs) == 0:
        print(f"❌ No images found in {data_dir}")
        return

    print(f"\n📂 Using {len(imgs)} images")
    data_dicts = [{"image": str(f)} for f in imgs]

    # Create dataset
    print("🔄 Creating SmartCacheDataset...")
    transforms = get_transforms()
    dataset = SmartCacheDataset(
        data=data_dicts,
        transform=transforms,
        cache_num=len(data_dicts) // 2,  # Cache 50%
        num_init_workers=2,
        num_replace_workers=1
    )

    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    # Model
    print("🏗️  Building model...")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=1
    ).to(device)

    criterion = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training
    print(f"\n🚀 Starting training: {epochs} epochs, batch={batch_size}")
    print(f"   Device: {device}")

    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(loader):
            x = batch["image"].to(device, non_blocking=True)
            y = (x > 0.5).float()

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                out = model(x)
                loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(loader)}, "
                      f"Loss: {loss.item():.4f}", end='\r')

        torch.cuda.synchronize()
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(loader)

        print(f"  Epoch {epoch+1}/{epochs} done: "
              f"Loss={avg_loss:.4f}, Time={epoch_time:.2f}s" + " "*30)

    total_time = time.time() - start_time

    # Report statistics
    print(f"\n{'='*70}")
    print("Training Complete")
    print(f"{'='*70}")

    gpu_mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    gpu_mem_reserved_mb = torch.cuda.memory_reserved(device) / 1024**2

    print(f"\n⏱️  Total time: {total_time:.2f}s")
    print(f"📊 Throughput: {len(imgs) * epochs / total_time:.2f} samples/s")
    print(f"\n💾 GPU Memory:")
    print(f"   Allocated: {gpu_mem_mb:.0f} MB")
    print(f"   Reserved: {gpu_mem_reserved_mb:.0f} MB")

    # CPU info
    cpu_percent = psutil.cpu_percent(interval=0.5)
    process = psutil.Process()
    process_mem_mb = process.memory_info().rss / 1024**2

    print(f"\n🧠 CPU:")
    print(f"   Load: {cpu_percent:.1f}%")
    print(f"   Process RAM: {process_mem_mb:.0f} MB")

    print(f"\n{'='*70}")

    return {
        'total_time': total_time,
        'gpu_mem_mb': gpu_mem_mb,
        'cpu_percent': cpu_percent
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--process_id", type=int, default=None)
    args = parser.parse_args()

    train_single_process(
        epochs=args.epochs,
        batch_size=args.batch_size,
        process_id=args.process_id
    )
PYEOF

chmod +x chapter3_maximise_GPU/train_single.py
print_success "Created: train_single.py"
echo ""

print_section "1.2: Parallel Training Orchestrator"

cat > chapter3_maximise_GPU/train_parallel.py << 'PYEOF'
#!/usr/bin/env python3
"""
Parallel Training Orchestrator
Manages multiple training processes
"""

import subprocess
import time
import argparse
import signal
import sys


class ParallelTrainer:
    def __init__(self, num_processes, epochs=3, batch_size=2):
        self.num_processes = num_processes
        self.epochs = epochs
        self.batch_size = batch_size
        self.processes = []

    def start(self):
        """Launch parallel training processes"""
        print("="*70)
        print(f"Launching {self.num_processes} Parallel Training Processes")
        print("="*70)

        for i in range(self.num_processes):
            print(f"\n🚀 Starting process {i+1}/{self.num_processes}...")

            cmd = [
                "python3",
                "chapter3_maximise_GPU/train_single.py",
                "--epochs", str(self.epochs),
                "--batch_size", str(self.batch_size),
                "--process_id", str(i+1)
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            self.processes.append({
                'id': i+1,
                'process': process,
                'start_time': time.time()
            })

            time.sleep(2)  # Stagger starts

        print(f"\n✓ All {self.num_processes} processes launched!")
        print("\n💡 Monitor with:")
        print("   watch -n 1 nvidia-smi")
        print("   htop")
        print("\nPress Ctrl+C to stop all processes")

    def monitor(self):
        """Monitor running processes"""
        try:
            while any(p['process'].poll() is None for p in self.processes):
                # Check for output from processes
                for p in self.processes:
                    if p['process'].poll() is None:
                        line = p['process'].stdout.readline()
                        if line:
                            print(f"[P{p['id']}] {line.strip()}")

                time.sleep(0.1)

            # All processes completed
            print("\n" + "="*70)
            print("All Training Processes Completed")
            print("="*70)

            for p in self.processes:
                elapsed = time.time() - p['start_time']
                print(f"  Process {p['id']}: {elapsed:.2f}s")

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted! Stopping all processes...")
            self.stop_all()

    def stop_all(self):
        """Stop all running processes"""
        for p in self.processes:
            if p['process'].poll() is None:
                print(f"  Stopping process {p['id']}...")
                p['process'].terminate()
                try:
                    p['process'].wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p['process'].kill()

        print("✓ All processes stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple training processes in parallel"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=2,
        help="Number of parallel processes"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Epochs per process"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size"
    )
    args = parser.parse_args()

    trainer = ParallelTrainer(
        num_processes=args.num_processes,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    trainer.start()
    trainer.monitor()


if __name__ == "__main__":
    main()
PYEOF

chmod +x chapter3_maximise_GPU/train_parallel.py
print_success "Created: train_parallel.py"
echo ""

print_section "1.3: Real-Time GPU Monitor"

cat > chapter3_maximise_GPU/monitor_gpu.py << 'PYEOF'
#!/usr/bin/env python3
"""
Real-time GPU Monitoring Dashboard
"""

import time
import os
import sys
try:
    import pynvml
except ImportError:
    print("❌ pynvml not installed: pip install pynvml")
    sys.exit(1)


def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')


def get_gpu_stats():
    """Get current GPU statistics"""
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    stats = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)

        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0

        # Get processes
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)

        stats.append({
            'id': i,
            'mem_used': mem_info.used / 1024**2,
            'mem_total': mem_info.total / 1024**2,
            'mem_percent': (mem_info.used / mem_info.total) * 100,
            'gpu_util': util.gpu,
            'mem_util': util.memory,
            'temp': temp,
            'power': power,
            'num_processes': len(processes)
        })

    pynvml.nvmlShutdown()
    return stats


def draw_bar(percent, width=30):
    """Draw a progress bar"""
    filled = int(width * percent / 100)
    bar = '█' * filled + '░' * (width - filled)
    return bar


def monitor_loop(interval=1):
    """Main monitoring loop"""
    try:
        while True:
            clear_screen()

            print("="*80)
            print(" "*20 + "🎮 REAL-TIME GPU MONITOR 🎮")
            print("="*80)
            print(f"Updates every {interval}s | Press Ctrl+C to exit\n")

            stats = get_gpu_stats()

            for gpu in stats:
                print(f"\n┌─ GPU {gpu['id']} ─────────────────────────────────────────────────────────┐")

                # Memory
                mem_bar = draw_bar(gpu['mem_percent'], width=40)
                print(f"│ Memory:  {mem_bar} {gpu['mem_percent']:.1f}%")
                print(f"│          {gpu['mem_used']:.0f} / {gpu['mem_total']:.0f} MB")

                # Utilization
                gpu_bar = draw_bar(gpu['gpu_util'], width=40)
                print(f"│ GPU:     {gpu_bar} {gpu['gpu_util']}%")

                mem_util_bar = draw_bar(gpu['mem_util'], width=40)
                print(f"│ Mem BW:  {mem_util_bar} {gpu['mem_util']}%")

                # Temperature & Power
                temp_color = ''
                if gpu['temp'] > 80:
                    temp_color = '🔥'
                elif gpu['temp'] > 70:
                    temp_color = '🌡️'
                else:
                    temp_color = '❄️'

                print(f"│ Temp:    {temp_color} {gpu['temp']}°C")
                print(f"│ Power:   ⚡ {gpu['power']:.1f} W")
                print(f"│ Processes: {gpu['num_processes']} running")

                # Status indicators
                print(f"│")
                if gpu['gpu_util'] > 80:
                    print(f"│ Status:  ✅ OPTIMAL - GPU well utilized")
                elif gpu['gpu_util'] > 50:
                    print(f"│ Status:  ⚠️  MODERATE - Can add more processes")
                else:
                    print(f"│ Status:  ❌ LOW - GPU underutilized!")

                if gpu['mem_percent'] > 95:
                    print(f"│ Memory:  🚨 CRITICAL - Near OOM!")
                elif gpu['mem_percent'] > 85:
                    print(f"│ Memory:  ⚠️  HIGH - Approaching limit")
                else:
                    print(f"│ Memory:  ✅ SAFE")

                print(f"└────────────────────────────────────────────────────────────────┘")

            print("\n" + "="*80)
            print("💡 TIP: Add more processes if GPU util < 70% and memory < 85%")
            print("="*80)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=1, help="Update interval (seconds)")
    args = parser.parse_args()

    monitor_loop(interval=args.interval)
PYEOF

chmod +x chapter3_maximise_GPU/monitor_gpu.py
print_success "Created: monitor_gpu.py"
echo ""

# ================================================================
# STEP 2: USAGE GUIDE
# ================================================================

print_header "Step 2: Usage Guide & Examples"

print_section "Example 1: Single Process Training"

cat << 'EOF'
Run a single training process to establish baseline:

  python3 chapter3_maximise_GPU/train_single.py --epochs 3 --batch_size 2

While running, open another terminal and monitor:

  watch -n 1 nvidia-smi

Expected:
  • GPU utilization: 40-60%
  • Memory: 3-5 GB
  • Conclusion: GPU not fully utilized!
EOF

pause

print_section "Example 2: Parallel Training (Manual)"

cat << 'EOF'
Launch multiple processes manually using &:

  # Terminal 1: Start monitoring
  python3 chapter3_maximise_GPU/monitor_gpu.py

  # Terminal 2: Launch parallel trainings
  python3 chapter3_maximise_GPU/train_single.py --epochs 3 &
  python3 chapter3_maximise_GPU/train_single.py --epochs 3 &
  python3 chapter3_maximise_GPU/train_single.py --epochs 3 &

  # Check background jobs
  jobs

  # Bring job to foreground
  fg %1

  # Kill all background jobs
  kill $(jobs -p)

Expected:
  • GPU utilization: 70-90%
  • Memory: 9-15 GB (3× single process)
  • Significantly better utilization!
EOF

pause

print_section "Example 3: Parallel Training (Automated)"

cat << 'EOF'
Use the orchestrator for automated parallel training:

  # Launch 3 parallel processes
  python3 chapter3_maximise_GPU/train_parallel.py --num_processes 3 --epochs 3

  # In another terminal, monitor
  python3 chapter3_maximise_GPU/monitor_gpu.py

The orchestrator will:
  ✓ Launch processes with staggered starts
  ✓ Monitor all processes
  ✓ Handle graceful shutdown (Ctrl+C)
  ✓ Report completion times
EOF

pause

print_section "Example 4: Finding Optimal Parallel Count"

cat << 'EOF'
Strategy to find the sweet spot:

1. Start with 1 process:
   python3 chapter3_maximise_GPU/train_single.py

2. Check GPU util and memory:
   nvidia-smi

3. If GPU util < 70% AND memory < 85%, add more:
   python3 chapter3_maximise_GPU/train_parallel.py --num_processes 2

4. Keep increasing until:
   • GPU utilization > 80%, OR
   • GPU memory > 90%, OR
   • CPU load > 80%

5. If you hit OOM, reduce batch size:
   python3 chapter3_maximise_GPU/train_parallel.py --num_processes 3 --batch_size 1

EXAMPLE PROGRESSION:

  Processes  GPU Util  GPU Mem   Action
  ─────────  ────────  ───────   ──────
  1          45%       4 GB      Add more ↑
  2          70%       8 GB      Add more ↑
  3          85%       12 GB     Good! ✓
  4          88%       16 GB     OOM! ✗

  → Optimal: 3 processes
EOF

pause

# ================================================================
# STEP 3: BEST PRACTICES
# ================================================================

print_header "Step 3: Best Practices & Troubleshooting"

cat << 'EOF'
══════════════════════════════════════════════════════════════════
                      BEST PRACTICES
══════════════════════════════════════════════════════════════════

1️⃣  ALWAYS MONITOR:
   • GPU: nvidia-smi or python3 monitor_gpu.py
   • CPU: htop
   • Never launch blindly!

2️⃣  START SMALL:
   • Begin with 2 processes
   • Increase gradually
   • Watch for diminishing returns

3️⃣  USE SMARTCACHEDATASET:
   • Keeps data in RAM
   • Avoids disk I/O bottleneck
   • Essential for parallel training

4️⃣  STAGGER PROCESS STARTS:
   • Don't launch all at once
   • Give 2-3 seconds between launches
   • Prevents initialization race conditions

5️⃣  SET RESOURCE LIMITS:
   • num_workers = 2-4 per process
   • cache_num = 50% of dataset
   • batch_size = small (1-4)

══════════════════════════════════════════════════════════════════
                    TROUBLESHOOTING
══════════════════════════════════════════════════════════════════

PROBLEM: Out of Memory (OOM)
SYMPTOMS: CUDA OOM error, process crashes
SOLUTIONS:
  • Reduce batch size: --batch_size 1
  • Reduce parallel count
  • Reduce cache_num in SmartCacheDataset
  • Use smaller model

PROBLEM: Low GPU Utilization despite multiple processes
SYMPTOMS: GPU util < 50% with 3+ processes
SOLUTIONS:
  • Check CPU load (may be CPU-bound)
  • Increase num_workers in DataLoader
  • Reduce data augmentation complexity
  • Check disk I/O (use cached dataset)

PROBLEM: Training slower with more processes
SYMPTOMS: More processes = lower throughput
SOLUTIONS:
  • CPU overload - reduce processes
  • Memory thrashing - reduce cache_num
  • Too many workers - reduce num_workers
  • GPU memory saturation - reduce batch size

PROBLEM: Processes hanging or freezing
SYMPTOMS: No progress, high CPU but no GPU activity
SOLUTIONS:
  • DataLoader deadlock - reduce num_workers
  • Disk I/O bottleneck - use SSD/NVMe
  • Insufficient shared memory - increase Docker shm
  • Kill and restart: kill $(jobs -p)

══════════════════════════════════════════════════════════════════
                    MONITORING CHECKLIST
══════════════════════════════════════════════════════════════════

Before adding more processes, verify:

☐ GPU utilization < 80%
☐ GPU memory < 85%
☐ CPU load < 80%
☐ RAM usage < 90%
☐ No disk I/O bottleneck (use iotop)
☐ All current processes running smoothly

If ALL checks pass → Add 1 more process
If ANY fails → Investigate before scaling
EOF

pause

# ================================================================
# FINAL SUMMARY
# ================================================================

print_header "Lesson Complete! 🎉"

cat << 'EOF'
══════════════════════════════════════════════════════════════════
                    📚 LESSON SUMMARY
══════════════════════════════════════════════════════════════════

YOU LEARNED:
✓ How to analyze system capacity
✓ Why single processes underutilize GPUs
✓ Parallel training strategies
✓ Real-time monitoring techniques
✓ Troubleshooting common issues

SCRIPTS CREATED:
✓ analyze_system.py - System capability analysis
✓ train_single.py - Single process training
✓ train_parallel.py - Parallel orchestrator
✓ monitor_gpu.py - Real-time GPU monitor

══════════════════════════════════════════════════════════════════
                    🚀 QUICK START COMMANDS
══════════════════════════════════════════════════════════════════

1. Analyze your system:
   python3 chapter3_maximise_GPU/analyze_system.py

2. Single training (baseline):
   python3 chapter3_maximise_GPU/train_single.py

3. Start monitoring:
   python3 chapter3_maximise_GPU/monitor_gpu.py

4. Parallel training:
   python3 chapter3_maximise_GPU/train_parallel.py --num_processes 3

5. Manual parallel (advanced):
   python3 chapter3_maximise_GPU/train_single.py &
   python3 chapter3_maximise_GPU/train_single.py &

══════════════════════════════════════════════════════════════════
                    💡 KEY TAKEAWAYS
══════════════════════════════════════════════════════════════════

1. Single processes often underutilize GPUs (40-60%)
2. Parallel training can achieve 80-95% utilization
3. Always monitor: GPU, CPU, RAM
4. Start with 2 processes, scale gradually
5. SmartCacheDataset is essential for efficiency
6. Balance: GPU memory vs CPU cores vs RAM

REMEMBER:
  More processes ≠ Always faster
  Find YOUR system's sweet spot through experimentation!

══════════════════════════════════════════════════════════════════
EOF

print_success "Lesson complete!"
print_info "All scripts ready in: chapter3_maximise_GPU/"
echo ""