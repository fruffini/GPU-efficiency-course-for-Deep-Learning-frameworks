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
