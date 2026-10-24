# cuda-perf-lab-3050
Learning GPU performance by profiling 15 kernels on RTX 3050 Laptop 6GB. Measuring register usage, shared memory, occupancy and latency with Nsight Compute. Naive vs tiled vs WMMA. All benchmarks on consumer hardware.

### Why I'm building this
I'm a BTech student with a 3050 laptop. I want to work on AI inference performance. Most tutorials stop at "make it coalesced". I'm trying to go one step deeper - why does cuBLAS still beat my best tiled WMMA kernel? This repo is me profiling that gap: registers, shared mem, occupancy, SASS.

Not claiming H100 numbers. Just learning to measure correctly on what I have.
