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
