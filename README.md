# ParallelCuda

A collection of C++/CUDA parallel-computing labs that explore GPU and CPU calculation patterns, host-device data passing, and different CUDA memory and reduction techniques. Each lab is a standalone CUDA program with its own Visual Studio solution.

## Overview

Every lab follows the same host/device structure: the host (`__host__`) allocates and fills data, copies it to the GPU with `cudaMemcpy`, launches one or more kernels (`__global__`), copies the results back, and reports timing using either `std::chrono` or CUDA events. The labs progressively demonstrate matrix math, real-time visualization, parallel reduction in shared memory, constant vs. global memory, atomics with `curand`, and a multi-layer reduction pipeline.

## Labs

| Lab | Topic | What it demonstrates |
| --- | --- | --- |
| [`1-lab`](1-lab/kernel.cu) | Matrix multiplication | A 2D-grid `matrixMult` kernel ([`kernel.cu`](1-lab/kernel.cu)) multiplies two 211×211 integer matrices read from a file, verifies the result against a CPU reference, writes the output to a file, and times the kernel with `std::chrono`. |
| [`2-lab`](2-lab/kernel.cu) | Animated function plot | A real-time GLUT/`CPUAnimBitmap` animation ([`kernel.cu`](2-lab/kernel.cu)) rendering the curve `tan(sin(x) + cos(x))`; thread/block dimensions are derived from the device's `maxThreadsPerBlock`, and each frame is rendered on the GPU and copied to the host bitmap. |
| [`3-lab`](3-lab/kernel.cu) | Parallel reduction in shared memory | Computes six harmonic-style sums concurrently ([`kernel.cu`](3-lab/kernel.cu)) using `__shared__` per-block caches and a tree reduction with `__syncthreads`, then finishes the block-level sums on the host. |
| [`4-lab`](4-lab/kernel.cu) | Constant vs. global memory | Runs the same array computation twice ([`kernel.cu`](4-lab/kernel.cu)), once reading coefficients from `__constant__` memory (`cudaMemcpyToSymbol`) and once from global memory, comparing their elapsed time via CUDA events. |
| [`5-lab`](5-lab/kernel.cu) | Monte Carlo Pi (atomics) | Estimates Pi by sampling random points with `curand` and counting hits with `atomicAdd` ([`kernel.cu`](5-lab/kernel.cu)), offering an optimized (count points outside the circle) and an unoptimized variant, timed with CUDA events. |
| [`control-task`](control-task/kernel.cu) | Multi-layer reduction pipeline | Passes a random array through a sequence of differently sized layers ([`kernel.cu`](control-task/kernel.cu)); each layer launches a kernel that applies `tanh` to a `curand`-weighted sum and feeds its output into the next layer, timed with CUDA events. |

## Tech stack

- **Language:** C++ with CUDA C (`.cu`)
- **Compiler/toolkit:** NVIDIA CUDA Toolkit 11.1 (`nvcc`), with the `curand` device API used in labs 5 and [`control-task`](control-task)
- **Build system:** Visual Studio 2019 solutions (`.sln`/`.vcxproj`, Platform Toolset v142, CUDA 11.1 build customizations)
- **Lab 2 extras:** OpenGL/GLUT helpers ([`gl_helper.h`](2-lab/gl_helper.h), [`cpu_anim.h`](2-lab/cpu_anim.h))

## Build and run

Each lab is an independent Visual Studio solution.

**Visual Studio (recommended on Windows):**

1. Open the lab's `.sln` (for example [`1-lab/1-lab.sln`](1-lab/1-lab.sln)) in Visual Studio 2019 with the CUDA 11.1 Toolkit installed.
2. Build the solution and run the resulting executable.

**Command line with `nvcc`:**

```sh
# from a lab directory, e.g. 1-lab
nvcc kernel.cu -o lab
./lab
```

Lab 1 prompts for input and output filenames at runtime and generates a random input file if needed. Lab 2 opens an animation window and requires OpenGL/GLUT and the [`gl_helper.h`](2-lab/gl_helper.h)/[`cpu_anim.h`](2-lab/cpu_anim.h) headers on the include path. The timing-comparison labs (4, 5, [`control-task`](control-task)) print their results and elapsed time to the console.

## Project structure

```
ParallelCuda/
├── 1-lab/            # Matrix multiplication
│   ├── kernel.cu
│   └── 1-lab.sln / 1-lab.vcxproj
├── 2-lab/            # Animated function plot (GLUT)
│   ├── kernel.cu
│   ├── cpu_anim.h
│   ├── gl_helper.h
│   └── 2-lab.sln / 2-lab.vcxproj
├── 3-lab/            # Parallel reduction in shared memory
│   ├── kernel.cu
│   └── 3-lab.sln / 3-lab.vcxproj
├── 4-lab/            # Constant vs. global memory
│   ├── kernel.cu
│   └── 4-lab.sln / 4-lab.vcxproj
├── 5-lab/            # Monte Carlo Pi with atomics
│   ├── kernel.cu
│   └── 5-lab.sln / 5-lab.vcxproj
└── control-task/     # Multi-layer reduction pipeline
    ├── kernel.cu
    └── control-work.sln / control-work.vcxproj
```
