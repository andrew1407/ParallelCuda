// cuRAND's default generator (curandState_t, XORWOW) in GLSL, the same port as curand.wgsl.
// curand_init(seed, 0, 0, &state): subsequence 0 and offset 0 need no skip-ahead, so each
// fragment builds the very state a CUDA thread would from the same seed. GLSL has no double,
// so curand_uniform_double's 53-bit fraction is rounded to f32.

struct curandState { uint d, v0, v1, v2, v3, v4; };

curandState curand_init(uint seed_lo, uint seed_hi) {
  uint s0 = seed_lo ^ 0xaad26b49u;
  uint s1 = seed_hi ^ 0xf7dcefddu;
  uint t0 = 1099087573u * s0;
  uint t1 = 2591861531u * s1;
  return curandState(6615241u + t1 + t0, 123456789u + t0, 362436069u ^ t0, 521288629u + t1, 88675123u ^ t1, 5783321u + t0);
}

uint curand(inout curandState state) {
  uint t = state.v0 ^ (state.v0 >> 2u);
  state.v0 = state.v1;
  state.v1 = state.v2;
  state.v2 = state.v3;
  state.v3 = state.v4;
  state.v4 = (state.v4 ^ (state.v4 << 4u)) ^ (t ^ (t << 1u));
  state.d += 362437u;
  return state.v4 + state.d;
}

// z = x ^ (y << 21) as a 53-bit integer, z / 2^53 + 2^-54
float curand_uniform_double(inout curandState state) {
  uint x = curand(state);
  uint y = curand(state);
  uint hi = y >> 11u;
  uint lo = x ^ (y << 21u);
  return float(hi) * 4.76837158203125e-7 + float(lo) * 1.1102230246251565e-16 + 5.551115123125783e-17;
}
