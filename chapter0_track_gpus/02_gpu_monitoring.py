#!/usr/bin/env python3
"""
Chapter 0: Real-time GPU Monitoring with Python
"""

import time
import sys
from datetime import datetime
from typing import List, Dict, Optional
import pynvml
import psutil


class GPUMonitor:
    """Monitor GPU metrics using NVIDIA Management Library"""

    def __init__(self):
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()
        print(f"✓ Initialized NVML. Found {self.device_count} GPU(s)")

    def get_gpu_info(self, device_id: int) -> Dict:
        """Get comprehensive GPU information"""
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)  # NVML handle for the GPU at index `device_id`
        name = pynvml.nvmlDeviceGetName(handle)  # GPU model/name (may be bytes on some bindings)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)  # memory stats object with total, free, used (bytes)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)  # utilization struct with `gpu` and `memory` percentages
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)  # GPU temperature in °C
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)  # current power draw in milliwatts
        power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)  # configured power limit in milliwatts
        sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)  # SM/core clock in MHz
        mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)  # memory clock in MHz

        return {
                'id'             : device_id,
                'name'           : name,
                'memory_used_mb' : mem_info.used / 1024 ** 2,
                'memory_total_mb': mem_info.total / 1024 ** 2,
                'memory_free_mb' : mem_info.free / 1024 ** 2,
                'memory_util_pct': (mem_info.used / mem_info.total) * 100,
                'gpu_util_pct'   : util.gpu,
                'temperature_c'  : temperature,
                'power_w'        : power_mw / 1000.0,
                'power_limit_w'  : power_limit_mw / 1000.0,
                'sm_clock_mhz'   : sm_clock,
                'mem_clock_mhz'  : mem_clock,
        }

    def get_gpu_processes(self, device_id: int) -> List[Dict]:
        """Get processes running on GPU"""
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        process_list = []

        for proc in processes:
            p = psutil.Process(proc.pid)
            proc_info = {
                    'pid'           : proc.pid,
                    'used_memory_mb': proc.usedGpuMemory / 1024 ** 2,
                    'name'          : p.name(),
                    'cmdline'       : ' '.join(p.cmdline()[:3])
            }
            process_list.append(proc_info)

        return process_list

    def print_summary(self):
        """Print formatted GPU summary"""
        print("\n" + "=" * 80)
        print(f"GPU Monitoring - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        for i in range(self.device_count):
            info = self.get_gpu_info(i)

            print(f"\n┌─ GPU {info['id']}: {info['name']}")
            print(f"├─ Memory: {info['memory_used_mb']:.0f} / {info['memory_total_mb']:.0f} MB "
                  f"({info['memory_util_pct']:.1f}%)")
            print(f"├─ Utilization: GPU {info['gpu_util_pct']}% | "
                  f"Memory {info['memory_util_pct']}%")
            print(f"├─ Temperature: {info['temperature_c']}°C")

            power_pct = (info['power_w'] / info['power_limit_w']) * 100
            print(f"├─ Power: {info['power_w']:.1f}W / {info['power_limit_w']:.1f}W "
                  f"({power_pct:.1f}%)")

            print(f"├─ Clocks: SM {info['sm_clock_mhz']} MHz | "
                  f"Memory {info['mem_clock_mhz']} MHz")

            processes = self.get_gpu_processes(i)
            if processes:
                print(f"└─ Processes ({len(processes)}):")
                for proc in processes:
                    print(f"   ├─ PID {proc['pid']}: {proc['name']} "
                          f"({proc['used_memory_mb']:.0f} MB)")
                    if proc['cmdline']:
                        print(f"   │  {proc['cmdline']}")
            else:
                print(f"└─ No processes running")

        print("\n" + "=" * 80)

    def monitor_loop(self, interval: int = 2, iterations: Optional[int] = None):
        """Continuous monitoring loop"""
        print(f"\nStarting GPU monitoring (interval: {interval}s)")
        print("Press Ctrl+C to stop\n")

        count = 0
        while True:
            self.print_summary()
            count += 1

            if iterations and count >= iterations:
                break

            time.sleep(interval)

    def check_gpu_health(self) -> bool:
        """Check for GPU health issues"""
        print("\n" + "=" * 80)
        print("GPU Health Check")
        print("=" * 80)

        all_healthy = True

        for i in range(self.device_count):
            info = self.get_gpu_info(i)
            issues = []

            if info['temperature_c'] > 85:
                issues.append(f"⚠️  High temperature: {info['temperature_c']}°C")
                all_healthy = False

            if info['memory_util_pct'] > 95:
                issues.append(f"⚠️  Memory near full: {info['memory_util_pct']:.1f}%")
                all_healthy = False

            if info['gpu_util_pct'] < 20 and info['memory_used_mb'] > 1000:
                issues.append(f"⚠️  Low GPU utilization ({info['gpu_util_pct']}%) "
                              f"despite allocated memory")
                all_healthy = False

            if issues:
                print(f"\nGPU {i}: ❌ ISSUES FOUND")
                for issue in issues:
                    print(f"  {issue}")
            else:
                print(f"\nGPU {i}: ✓ Healthy")

        print("\n" + "=" * 80)
        return all_healthy

    def cleanup(self):
        """Cleanup NVML"""
        pynvml.nvmlShutdown()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='GPU Monitoring Tool')
    parser.add_argument('--interval', type=int, default=2, help='Monitoring interval in seconds')
    parser.add_argument('--iterations', type=int, default=None, help='Number of iterations')
    parser.add_argument('--once', action='store_true', help='Print summary once and exit')
    parser.add_argument('--health-check', action='store_true', help='Run health check and exit')
    args = parser.parse_args()

    monitor = GPUMonitor()

    if args.health_check:
        healthy = monitor.check_gpu_health()
        monitor.cleanup()
        sys.exit(0 if healthy else 1)
    elif args.once:
        monitor.print_summary()
        monitor.cleanup()
    else:
        monitor.monitor_loop(interval=args.interval, iterations=args.iterations)
        monitor.cleanup()


if __name__ == '__main__':
    main()