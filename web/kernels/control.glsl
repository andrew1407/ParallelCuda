// WebGL2 port of kernel from control-task/kernel.cu: one layer of the tanh pipeline, one fragment
// per output (the lab's <<<outputLength, 1>>>). As in control.wgsl, the host's double sum arrives as
// its raw 64 bits: truncated exactly for the curand seed, (unsigned long long)*sum, and rounded to
// f32 for the math. Output i sits at (i % W, i / W) of an R32F texture.

//!include curand.glsl

uniform uvec2 sum; // (low word, high word) of the double
uniform int W;

out float layerOutput;

// (unsigned long long)d for 0 <= d < 2^32: the integer part of the double's bits
uint f64_trunc_u32(uvec2 bits) {
  int e = int((bits.y >> 20u) & 0x7ffu) - 1023;
  if (e < 0) return 0u;
  if (e > 31) return 0xffffffffu;
  uint shift = uint(52 - e); // 21 ..= 52
  uint mhi = (bits.y & 0xfffffu) | 0x100000u;
  if (shift >= 32u) return mhi >> (shift - 32u);
  return (bits.x >> shift) | (mhi << (32u - shift));
}

// the double rounded down to f32: same sign, rebiased exponent, top 23 mantissa bits
float f64_to_f32(uvec2 bits) {
  int e = int((bits.y >> 20u) & 0x7ffu) - 1023;
  if (e < -126) return 0.0;
  uint mant = ((bits.y & 0xfffffu) << 3u) | (bits.x >> 29u);
  return uintBitsToFloat((bits.y & 0x80000000u) | (uint(e + 127) << 23u) | mant);
}

float func(float x) {
  // exp overflows f32 above 88; tanh is exactly ±1 in f32 well before 44, so clamp first
  float xc = clamp(x, -44.0, 44.0);
  float expPlus = exp(xc);
  float expMinus = exp(-xc);
  return (expPlus - expMinus) / (expPlus + expMinus);
}

void main() {
  uint i = uint(gl_FragCoord.x) + uint(gl_FragCoord.y) * uint(W); // threadIdx.x + blockIdx.x * blockDim.x
  curandState state = curand_init(f64_trunc_u32(sum) + i, 0u); // (unsigned long long)*sum + i
  float weight = curand_uniform_double(state);
  float a = weight * f64_to_f32(sum) + 1.0;
  layerOutput = func(a);
}
