// WebGPU port of matrixMult from 1-lab/kernel.cu.
// The lab launches matrixMult<<<dim3(MATRIX_SIZE, MATRIX_SIZE), 1>>>: one workgroup per
// result element, one invocation each. BLOCK > 1 tries square blocks instead.

override MATRIX_SIZE: u32 = 211u;
override BLOCK: u32 = 1u;

@group(0) @binding(0) var<storage, read> matrixA: array<i32>;
@group(0) @binding(1) var<storage, read> matrixB: array<i32>;
@group(0) @binding(2) var<storage, read_write> matrixResult: array<i32>;

@compute @workgroup_size(BLOCK, BLOCK, 1)
fn matrixMult(
  @builtin(workgroup_id) blockIdx: vec3<u32>,
  @builtin(local_invocation_id) threadIdx: vec3<u32>,
) {
  let blockDim = vec3<u32>(BLOCK, BLOCK, 1u);
  let MATRIX_LENGTH = MATRIX_SIZE * MATRIX_SIZE;
  let i = MATRIX_SIZE * (blockDim.y * blockIdx.y + threadIdx.y);
  let j = blockDim.x * blockIdx.x + threadIdx.x;
  let resIndex = MATRIX_SIZE * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
  // the lab's bounds check, kept as is: j is compared with MATRIX_LENGTH, not MATRIX_SIZE
  if (i >= MATRIX_LENGTH || j >= MATRIX_LENGTH || resIndex >= MATRIX_LENGTH) {
    return;
  }
  var resValue: i32 = 0;
  for (var k = 0u; k < MATRIX_SIZE; k++) {
    resValue += matrixA[i + k] * matrixB[k * MATRIX_SIZE + j];
  }
  matrixResult[resIndex] = resValue;
}
