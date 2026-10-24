# cuda-perf-lab-3050
Learning GPU performance by profiling 15 kernels on RTX 3050 Laptop 6GB. Measuring register usage, shared memory, occupancy and latency with Nsight Compute. Naive vs tiled vs WMMA. All benchmarks on consumer hardware.

### Why I'm building this
I'm a BTech student with a 3050 laptop. I want to work on AI inference performance. Most tutorials stop at "make it coalesced". I'm trying to go one step deeper - why does cuBLAS still beat my best tiled WMMA kernel? This repo is me profiling that gap: registers, shared mem, occupancy, SASS.

Not claiming H100 numbers. Just learning to measure correctly on what I have.

# cuda-perf-lab-3050

Learning CUDA performance optimization on a RTX 3050 Laptop 6GB.

This repo is my notes from building 15 kernels from scratch and profiling them with Nsight Compute. Focus is on understanding why kernels are slow and what actually speeds them up on consumer hardware.

**Hardware:**
- RTX 3050 Laptop 6GB, sm_86
- 224 GB/s bandwidth

**What I'm measuring for each kernel:**
- Latency, GB/s, Occupancy, Registers, Spill
- Nsight Compute screenshots in /profiles

**Constraints I follow:**
- 1024x1024 for matmul to fit in 6GB
- FP16 where possible
- Try to keep registers <= 32 and spill = 0 using __launch_bounds__

**Progress - MatMul 1024 FP16:**
**Progress - MatMul 1024 FP16:**

| Version | Status | Notes |
|---|---|---|
| naive | done | 9.33ms, 27 regs, 0 spill - measured on 3050 6GB |
| tiled 32x32 | todo | Shared memory |
| wmma | todo | Tensor cores |
| cuBLAS | todo | Baseline |

> Will fill after profiling on my 3050. No estimated numbers.

Goal is to understand the gap between my wmma and cuBLAS by looking at PTX/SASS.

### Structure
- kernels/01_matmul/ - code
- profiles/ - ncu reports
- bench/ - benchmarking scripts

Work in progress.
