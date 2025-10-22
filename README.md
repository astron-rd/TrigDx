# TrigDx

High‑performance C++ library that provides multiple implementations of transcendental trigonometric functions (sin, cos, tan and variant forms). TrigDx is designed for numerical, signal-processing, and real‑time systems where you can trade a small amount of accuracy for significantly higher throughput on modern CPUs (scalar & SIMD) and NVIDIA GPUs.

## Why TrigDx?
Many applications use the standard library implementations, which prioritise correctness but are not always optimal for throughput on vectorized or GPU hardware. TrigDx gives you multiple implementations so you can:

- Replace `std::sin` / `std::cos` calls with faster approximations when a small, bounded reduction in accuracy is acceptable.
- Use SIMD/vectorized implementations and compact lookup tables for high throughput lookups.
- Run massively parallel kernels that take advantage of GPU' _Special Function Units_ (SFUs)

## Requirements
- A C++ compiler with at least C++17 support (GCC, Clang)
- CMake 3.15+
- Optional: NVIDIA CUDA Toolkit 11+ to build GPU kernels
- Optional: GoogleTest (for unit tests) and GoogleBenchmark (for microbenchmarks)

## Building
```bash
git clone https://github.com/astron-rd/TrigDx.git
cd TrigDx
mkdir build && cd build

# CPU-only:
cmake -DCMAKE_BUILD_TYPE=Release -DTRIGDX_USE_XSIMD=ON ..
cmake --build . -j

# Enable CUDA (if available):
cmake -DCMAKE_BUILD_TYPE=Release -DTRIGDX_USE_GPU=ON ..
cmake --build . -j

# Run tests:
ctest --output-on-failure -j
```

CMake options (common)
- `TRIGDX_USE_GPU=ON/OFF` — build GPU support
- `TRIGDX_BUILD_TESTS=ON/OFF` — build tests
- `TRIGDX_BUILD_BENCHMARKS=ON/OFF` — build benchmarks
- `TRIGDX_BUILD_PYTHON` — build Python interface

## Contributing
- Fork → create a feature branch → open a PR.
- Include unit tests for correctness‑sensitive changes and benchmark results for performance changes.
- Follow project style (clang‑format) and run tests locally before submitting.

## Reporting issues
When opening an issue for incorrect results or performance regressions, please include:
- Platform and CPU/GPU model
- Compiler and version with exact compile flags
- Small reproducer (input data and the TrigDx implementation used)
  
## License
See the LICENSE file in the repository for licensing details.
