// WebGL2 port of calculateSums from 3-lab/kernel.cu. A fragment shader has no __shared__ memory
// and no __syncthreads(), so the per-block cache is a THREADS_PER_BLOCK × BLOCKS float texture
// (row = block, column = threadIdx.x, two RGBA32F render targets for the six sums), and each
// step of the tree reduction is a pass of its own: the end of a pass is the barrier.
// INIT fills the cache like the kernel's first half; REDUCE does cache[t] += cache[t + i] for
// t < i, the same additions in the same order, so the f32 sums match the WebGPU port.

layout(location = 0) out vec4 cache0; // A, B, C1, C2
layout(location = 1) out vec4 cache1; // D1, D2

#ifdef INIT
uniform uint THREADS_PER_BLOCK;
uniform uint ARRAY_LENGTH;

void main() {
  uint threadIdx = uint(gl_FragCoord.x), blockIdx = uint(gl_FragCoord.y);
  uint valIdx = threadIdx + blockIdx * THREADS_PER_BLOCK;
  // threads past the array add zeros
  bool inRange = valIdx < ARRAY_LENGTH;
  uint val = valIdx + 1u;
  float res1 = inRange ? 1.0 / float(val) : 0.0;
  float res2 = inRange ? 1.0 / float(ARRAY_LENGTH - valIdx) : 0.0;
  if (val % 2u != 0u) {
    cache0 = vec4(res1, res2, 0.0, res1);
    cache1 = vec4(0.0, res2, 0.0, 0.0);
  } else {
    cache0 = vec4(res1, res2, res1, 0.0);
    cache1 = vec4(res2, 0.0, 0.0, 0.0);
  }
}
#endif

#ifdef REDUCE
uniform sampler2D src0;
uniform sampler2D src1;
uniform int i; // the half-width of this step: THREADS_PER_BLOCK / 2, / 4, … 1

void main() {
  ivec2 p = ivec2(gl_FragCoord.xy), q = p + ivec2(i, 0); // cacheIdx and cacheIdx + i
  cache0 = texelFetch(src0, p, 0) + texelFetch(src0, q, 0);
  cache1 = texelFetch(src1, p, 0) + texelFetch(src1, q, 0);
}
#endif
