// WebGPU port of kernel from control-task/kernel.cu: one layer of the tanh pipeline,
// launched <<<outputLength, 1>>> as dispatchWorkgroups(outputLength) of one invocation.
// The host sum is a double. WGSL has no f64, so the kernel gets its raw 64 bits: it truncates
// them exactly for the curand seed, (unsigned long long)*sum, and rounds them to f32 for the math.

//!include curand.wgsl

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> sum: vec2<u32>; // (low word, high word) of the double

// (unsigned long long)d for 0 <= d < 2^32: the integer part of the double's bits
fn f64_trunc_u32(bits: vec2<u32>) -> u32 {
  let e = i32((bits.y >> 20u) & 0x7ffu) - 1023;
  if (e < 0) {
    return 0u;
  }
  if (e > 31) {
    return 0xffffffffu;
  }
  let shift = u32(52 - e); // 21 ..= 52
  let mhi = (bits.y & 0xfffffu) | 0x100000u;
  if (shift >= 32u) {
    return mhi >> (shift - 32u);
  }
  return (bits.x >> shift) | (mhi << (32u - shift));
}

// the double rounded down to f32: same sign, rebiased exponent, top 23 mantissa bits
fn f64_to_f32(bits: vec2<u32>) -> f32 {
  let e = i32((bits.y >> 20u) & 0x7ffu) - 1023;
  if (e < -126) {
    return 0.0;
  }
  let mant = ((bits.y & 0xfffffu) << 3u) | (bits.x >> 29u);
  return bitcast<f32>((bits.y & 0x80000000u) | (u32(e + 127) << 23u) | mant);
}

fn func(x: f32) -> f32 {
  // exp overflows f32 above 88; tanh is exactly ±1 in f32 well before 44, so clamp first
  let xc = clamp(x, -44.0, 44.0);
  let expPlus = exp(xc);
  let expMinus = exp(-xc);
  return (expPlus - expMinus) / (expPlus + expMinus);
}

@compute @workgroup_size(1, 1, 1)
fn layerKernel(
  @builtin(workgroup_id) blockIdx: vec3<u32>,
  @builtin(local_invocation_id) threadIdx: vec3<u32>,
) {
  let blockDim = 1u;
  let i = threadIdx.x + blockIdx.x * blockDim;
  var state = curand_init(f64_trunc_u32(sum), 0u);
  let weight = curand_uniform_double(&state);
  let a = weight * f64_to_f32(sum) + 1.0;
  output[i] = func(a);
}
