// WebGPU port of kernelGlobal and kernelConst from 4-lab/kernel.cu.
// __constant__ float constArrConst[CONST_SIZE] becomes a uniform buffer; the global-memory copy is a
// read-only storage buffer. The lab launches <<<1, ARR_SIZE>>>; when the device allows fewer
// invocations per workgroup, the same threads are split over ARR_SIZE / WG workgroups.

override ARR_SIZE: u32 = 512u;
override WG: u32 = 512u;

// uniform arrays need 16-byte elements, so the floats are packed four to a vec4:
// 5 × 4 = 20 floats hold the CONST_SIZE = 17 coefficients.
struct Constants { v: array<vec4<f32>, 5> }

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read> constArr: array<f32>;
@group(0) @binding(3) var<uniform> constArrConst: Constants;

fn constAt(u: u32) -> f32 {
  return constArrConst.v[u / 4u][u % 4u];
}

@compute @workgroup_size(WG, 1, 1)
fn kernelGlobal(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; // threadIdx.x + blockIdx.x * blockDim.x
  if (i >= ARR_SIZE) {
    return;
  }
  let fi = f32(i);
  B[i] = 0.0;
  if (i == 0u) {
    return; // the formula divides by i, so B[0] stays 0
  }
  for (var u = 1u; u <= 15u; u += 2u) {
    B[i] += select(
      constArr[0] * A[i] * fi + constArr[15] * A[i] / fi,
      constArr[u] * A[i] / fi - constArr[u + 1u] * A[i] * fi,
      i != 15u,
    );
  }
}

@compute @workgroup_size(WG, 1, 1)
fn kernelConst(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ARR_SIZE) {
    return;
  }
  let fi = f32(i);
  B[i] = 0.0;
  if (i == 0u) {
    return; // the formula divides by i, so B[0] stays 0
  }
  for (var u = 1u; u <= 15u; u += 2u) {
    B[i] += select(
      constAt(0u) * A[i] * fi + constAt(15u) * A[i] / fi,
      constAt(u) * A[i] / fi - constAt(u + 1u) * A[i] * fi,
      i != 15u,
    );
  }
}
