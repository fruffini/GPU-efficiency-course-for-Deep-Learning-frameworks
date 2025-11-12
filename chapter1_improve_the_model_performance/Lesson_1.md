# Chapter 1: Improve Model Performance

## Learning Objectives

By the end of this chapter, you will:
- Optimize GPU memory utilization through batch size and volume scaling
- Apply mixed precision training (AMP, BF16, FP16) for 2-3× speedups
- Use gradient accumulation to simulate large batches on limited memory
- Calculate and maximize GPU FLOPS (floating-point operations per second)
- Measure model efficiency and identify optimization opportunities
- Choose the best precision/accumulation combination for your hardware

## Prerequisites

- Completed Chapter 0 (GPU monitoring basics)
- PyTorch 2.0+ with CUDA support
- MONAI installed
- GPU with at least 16GB memory

## Quick Start
```bash
# Create directories
mkdir -p chapter1_improve_the_model_performance/figures
mkdir -p chapter1_improve_the_model_performance/results

# Exercise 5: Basic optimization strategies
python chapter1_improve_the_model_performance/mock_training_opt.py --batch_size 4 --amp

# Exercise 6: Scaling benchmarks
python chapter1_improve_the_model_performance/mock_training_scaling.py

# Exercise 7: Precision comparison
python chapter1_improve_the_model_performance/mock_precision_benchmark.py

# Exercise 8: Gradient accumulation
python chapter1_improve_the_model_performance/mock_precision_accumulation.py

# Exercise 9: FLOPS analysis
python chapter1_improve_the_model_performance/compute_flops_efficiency.py

# Exercise 10: Efficiency maximization
python chapter1_improve_the_model_performance/maximize_efficiency.py
```

## Key Metrics

### FLOPS (Floating-Point Operations Per Second)
- **TFLOPS**: Trillions of FLOPS
- **GPU Utilization**: Achieved FLOPS / Peak FLOPS × 100%
- **Target**: >50% utilization for training, >70% for inference

### Efficiency Metrics
- **MFU (Model FLOPS Utilization)**: Actual compute / Theoretical peak
- **Memory Bandwidth Utilization**: Actual bandwidth / Peak bandwidth
- **Samples per Second**: Higher = better throughput

## Expected Results

| Configuration | Memory | Throughput | TFLOPS | GPU Util |
|--------------|--------|------------|--------|----------|
| FP32, batch=1 | 6.2 GB | 2.1 samp/s | 8.4 | 34% |
| AMP, batch=2 | 4.8 GB | 4.8 samp/s | 15.2 | 61% |
| AMP, batch=4 | 8.1 GB | 6.3 samp/s | 19.8 | 79% |
| BF16, batch=4 | 7.9 GB | 6.8 samp/s | 21.1 | 84% |

*Benchmarks on A100 40GB*

## Files

1. **mock_training_opt.py**: Basic optimization (Exercise 5)
2. **mock_training_scaling.py**: Batch/volume scaling (Exercise 6)
3. **mock_precision_benchmark.py**: Precision comparison (Exercise 7)
4. **mock_precision_accumulation.py**: Gradient accumulation (Exercise 8)
5. **compute_flops_efficiency.py**: FLOPS analysis (Exercise 9)
6. **maximize_efficiency.py**: Complete optimization (Exercise 10)