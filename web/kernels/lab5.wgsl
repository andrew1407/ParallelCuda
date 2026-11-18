// WebGPU port of kernelOptimized and kernelUnoptimized from 5-lab/kernel.cu: Monte Carlo Pi with
// one curand state per point and an atomic counter (atomicAdd on an atomic<u32>).
// <<<BLOCKS, THREADS>>> is dispatchWorkgroups(BLOCKS) of @workgroup_size(THREADS).

//!include curand.wgsl

override ARR_LENGTH: u32 = 80000u;
override THREADS: u32 = 256u;

struct coords { x: f32, y: f32 }

@group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;

fn getCoordsByDist(i: u32) -> coords {
  var state = curand_init(i, 0u);
  let x = curand_uniform_double(&state);
  let y = curand_uniform_double(&state);
  return coords(x, y);
}

fn kernelExec(threadIdx: u32, blockIdx: u32, gridDim: u32, reversed: bool) {
  let blockDim = THREADS;
  var i = threadIdx + blockIdx * blockDim;
  let blockSize = blockDim * gridDim;
  while (i < ARR_LENGTH) {
    let dot = getCoordsByDist(i);
    let radius = dot.x * dot.x + dot.y * dot.y; // pow(dot.x, 2) + pow(dot.y, 2)
    let matches = select(radius <= 1.0, radius > 1.0, reversed);
    if (matches) {
      atomicAdd(&counter, 1u);
    }
    i += blockSize;
  }
}

@compute @workgroup_size(THREADS, 1, 1)
fn kernelOptimized(
  @builtin(local_invocation_id) threadIdx: vec3<u32>,
  @builtin(workgroup_id) blockIdx: vec3<u32>,
  @builtin(num_workgroups) gridDim: vec3<u32>,
) {
  kernelExec(threadIdx.x, blockIdx.x, gridDim.x, true);
}

@compute @workgroup_size(THREADS, 1, 1)
fn kernelUnoptimized(
  @builtin(local_invocation_id) threadIdx: vec3<u32>,
  @builtin(workgroup_id) blockIdx: vec3<u32>,
  @builtin(num_workgroups) gridDim: vec3<u32>,
) {
  kernelExec(threadIdx.x, blockIdx.x, gridDim.x, false);
}
