// WebGL2 port of kernelOptimized (REVERSED 1) and kernelUnoptimized (REVERSED 0) from 5-lab/kernel.cu.
// A fragment is one CUDA thread of the <<<BLOCKS, THREADS>>> grid and runs the same grid-stride loop,
// with one curand state per point seeded by its index. A fragment shader has no atomics, so instead
// of atomicAdd on one counter each thread writes how many of its points matched, and reduce.glsl
// adds those counts up in passes. The sum is exact (u32), so the count is the same as the atomics'.

//!include curand.glsl

uniform uint ARR_LENGTH;
uniform uint blockSize; // blockDim.x * gridDim.x
uniform uint W;

out uint counter;

void main() {
  uint threadId = uint(gl_FragCoord.x) + uint(gl_FragCoord.y) * W; // threadIdx.x + blockIdx.x * blockDim.x
  uint count = 0u;
  for (uint i = threadId; threadId < blockSize && i < ARR_LENGTH; i += blockSize) {
    curandState state = curand_init(i, 0u);
    float x = curand_uniform_double(state);
    float y = curand_uniform_double(state);
    float radius = x * x + y * y; // pow(dot.x, 2) + pow(dot.y, 2)
#if REVERSED
    if (radius > 1.0) count++;
#else
    if (radius <= 1.0) count++;
#endif
  }
  counter = count;
}
