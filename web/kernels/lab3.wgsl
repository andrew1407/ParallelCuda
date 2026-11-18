// WebGPU port of calculateSums from 3-lab/kernel.cu: six sums reduced per block in shared
// memory (var<workgroup>) with a tree reduction between barriers (workgroupBarrier).
// A workgroup is one CUDA block of THREADS_PER_BLOCK threads. If the device allows fewer
// invocations per workgroup (WG), each invocation plays THREADS_PER_BLOCK / WG of the threads;
// with WG = THREADS_PER_BLOCK it is one thread each, exactly like the lab. f32 replaces double.

override THREADS_PER_BLOCK: u32 = 1024u;
override WG: u32 = 1024u;
override ARRAY_LENGTH: u32 = 10000u;
override CACHE_LENGTH: u32 = 6u * THREADS_PER_BLOCK;
const sumsAmount = 6u;

@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C1: array<f32>;
@group(0) @binding(3) var<storage, read_write> C2: array<f32>;
@group(0) @binding(4) var<storage, read_write> D1: array<f32>;
@group(0) @binding(5) var<storage, read_write> D2: array<f32>;

// __shared__ baseType sumsCache[sumsAmount][THREADS_PER_BLOCK], flattened
var<workgroup> sumsCache: array<f32, CACHE_LENGTH>;

fn cache(u: u32, idx: u32) -> u32 {
  return u * THREADS_PER_BLOCK + idx;
}

@compute @workgroup_size(WG, 1, 1)
fn calculateSums(
  @builtin(workgroup_id) blockIdx: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  for (var threadIdx = lid.x; threadIdx < THREADS_PER_BLOCK; threadIdx += WG) {
    let valIdx = threadIdx + blockIdx.x * THREADS_PER_BLOCK;
    let cacheIdx = threadIdx;
    // The lab returns here, before __syncthreads, and leaves these cache slots unset.
    // Barriers must stay in uniform control flow in WGSL, so the thread writes zeros and stays.
    if (valIdx >= ARRAY_LENGTH) {
      for (var u = 0u; u < sumsAmount; u++) {
        sumsCache[cache(u, cacheIdx)] = 0.0;
      }
      continue;
    }
    let val = valIdx + 1u;
    let res1 = 1.0 / f32(val);
    let res2 = 1.0 / f32(ARRAY_LENGTH - valIdx);

    sumsCache[cache(0u, cacheIdx)] = res1;
    sumsCache[cache(1u, cacheIdx)] = res2;
    if (val % 2u != 0u) {
      sumsCache[cache(2u, cacheIdx)] = 0.0;
      sumsCache[cache(4u, cacheIdx)] = 0.0;
      sumsCache[cache(3u, cacheIdx)] = res1;
      sumsCache[cache(5u, cacheIdx)] = res2;
    } else {
      sumsCache[cache(3u, cacheIdx)] = 0.0;
      sumsCache[cache(5u, cacheIdx)] = 0.0;
      sumsCache[cache(2u, cacheIdx)] = res1;
      sumsCache[cache(4u, cacheIdx)] = res2;
    }
  }

  workgroupBarrier(); // __syncthreads()

  for (var i = THREADS_PER_BLOCK / 2u; i != 0u; i /= 2u) {
    for (var cacheIdx = lid.x; cacheIdx < i; cacheIdx += WG) {
      for (var u = 0u; u < sumsAmount; u++) {
        sumsCache[cache(u, cacheIdx)] += sumsCache[cache(u, cacheIdx + i)];
      }
    }
    workgroupBarrier(); // __syncthreads()
  }

  if (lid.x != 0u) {
    return;
  }
  A[blockIdx.x] = sumsCache[cache(0u, 0u)];
  B[blockIdx.x] = sumsCache[cache(1u, 0u)];
  C1[blockIdx.x] = sumsCache[cache(2u, 0u)];
  C2[blockIdx.x] = sumsCache[cache(3u, 0u)];
  D1[blockIdx.x] = sumsCache[cache(4u, 0u)];
  D2[blockIdx.x] = sumsCache[cache(5u, 0u)];
}
