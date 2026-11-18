// cuRAND's default generator (curandState_t, XORWOW), ported from curand_kernel.h.
// curand_init(seed, 0, 0, &state): subsequence 0 and offset 0 need no skip-ahead, so the
// state is the same as on the GPU. curand_uniform_double builds a 53-bit fraction from two
// draws; WGSL has no f64, so it is rounded to f32 here.

struct curandState {
  d: u32,
  v0: u32, v1: u32, v2: u32, v3: u32, v4: u32,
}

fn curand_init(seed_lo: u32, seed_hi: u32) -> curandState {
  let s0 = seed_lo ^ 0xaad26b49u;
  let s1 = seed_hi ^ 0xf7dcefddu;
  let t0 = 1099087573u * s0;
  let t1 = 2591861531u * s1;
  return curandState(6615241u + t1 + t0, 123456789u + t0, 362436069u ^ t0, 521288629u + t1, 88675123u ^ t1, 5783321u + t0);
}

fn curand(state: ptr<function, curandState>) -> u32 {
  let t = (*state).v0 ^ ((*state).v0 >> 2u);
  (*state).v0 = (*state).v1;
  (*state).v1 = (*state).v2;
  (*state).v2 = (*state).v3;
  (*state).v3 = (*state).v4;
  (*state).v4 = ((*state).v4 ^ ((*state).v4 << 4u)) ^ (t ^ (t << 1u));
  (*state).d += 362437u;
  return (*state).v4 + (*state).d;
}

// z = x ^ (y << 21) as a 53-bit integer, z / 2^53 + 2^-54
fn curand_uniform_double(state: ptr<function, curandState>) -> f32 {
  let x = curand(state);
  let y = curand(state);
  let hi = y >> 11u;
  let lo = x ^ (y << 21u);
  return f32(hi) * 4.76837158203125e-7 + f32(lo) * 1.1102230246251565e-16 + 5.551115123125783e-17;
}
