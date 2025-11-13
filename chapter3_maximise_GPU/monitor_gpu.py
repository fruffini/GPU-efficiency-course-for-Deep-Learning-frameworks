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
