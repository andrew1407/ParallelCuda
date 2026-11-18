// CPU references for the ParallelCuda labs, in plain JavaScript (f64, like the labs' host code
// and their double kernels). They do what each kernel does, thread by thread, so the WebGPU
// results can be checked against them and timed against them.

// ------------------------------------------------------------------ random --
// MSVC's rand(): the labs build with Visual Studio 2019 and never call srand(), so the seed is 1.
export const RAND_MAX = 0x7fff;
export function msvcRand(seed = 1) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 214013) + 2531011) >>> 0;
    return (s >>> 16) & RAND_MAX;
  };
}

// cuRAND's default generator (curandState_t = XORWOW), after curand_kernel.h. The labs call
// curand_init(seed, 0, 0, &state): subsequence 0 and offset 0 need no skip-ahead.
export function curandInit(seedLo, seedHi = 0) {
  const s0 = (seedLo ^ 0xaad26b49) >>> 0;
  const s1 = (seedHi ^ 0xf7dcefdd) >>> 0;
  const t0 = Math.imul(1099087573, s0) >>> 0;
  const t1 = Math.imul(2591861531, s1) >>> 0;
  return {
    d: (6615241 + t1 + t0) >>> 0,
    v: [(123456789 + t0) >>> 0, (362436069 ^ t0) >>> 0, (521288629 + t1) >>> 0, (88675123 ^ t1) >>> 0, (5783321 + t0) >>> 0],
  };
}
export function curand(st) {
  const v = st.v;
  const t = (v[0] ^ (v[0] >>> 2)) >>> 0;
  v[0] = v[1]; v[1] = v[2]; v[2] = v[3]; v[3] = v[4];
  v[4] = ((v[4] ^ (v[4] << 4)) ^ (t ^ (t << 1))) >>> 0;
  st.d = (st.d + 362437) >>> 0;
  return (v[4] + st.d) >>> 0;
}
// _curand_uniform_double_hq: z = x ^ (y << 21) as a 53-bit integer, then z / 2^53 + 2^-54
export function curandUniformDouble(st) {
  const x = curand(st), y = curand(st);
  const hi = y >>> 11, lo = (x ^ (y << 21)) >>> 0;
  return (hi * 4294967296 + lo) * 2 ** -53 + 2 ** -54;
}
// (unsigned long long)d for a non-negative double, split into the two 32-bit seed halves
const seedOf = (d) => { const t = Math.trunc(d); return [t % 4294967296 >>> 0, Math.floor(t / 4294967296) >>> 0]; };

// ------------------------------------------------------------------- lab 1 --
export function lab1Input(size, seed) {
  const rand = msvcRand(seed), n = size * size;
  const A = new Int32Array(n), B = new Int32Array(n);
  for (const m of [A, B]) for (let i = 0; i < n; i++) m[i] = rand() % 1000;
  return { A, B };
}
export function lab1Cpu(A, B, size) {
  const C = new Int32Array(size * size);
  for (let i = 0; i < size; i++)
    for (let j = 0; j < size; j++) {
      let sum = 0;
      for (let k = 0; k < size; k++) sum = (sum + Math.imul(A[i * size + k], B[k * size + j])) | 0;
      C[i * size + j] = sum;
    }
  return C;
}

// ------------------------------------------------------------------- lab 2 --
// p: { W, H, MAXX, MAXY, MINY, DT }; writes RGBA into a Uint32Array of W*H, row 0 = bottom row
export function lab2Frame(px, ticks, p) {
  const { W, H, MAXX, MAXY, MINY, DT } = p;
  const scale = Math.trunc(H / (MAXY - MINY)); // integer division, as in the lab's macros
  const span = MAXX + ticks * DT;
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      let hit = false;
      for (let xDelta = -0.5; xDelta <= 0.5; xDelta += 0.5) {
        const cX = (x + xDelta) / W * span;
        const c = Math.abs(scale * (Math.tan(Math.sin(cX) + Math.cos(cX)) - MINY) - y);
        hit = hit || c <= 1;
      }
      px[x + y * W] = hit ? 0xff0000ff : 0xffffffff;
    }
}

// ------------------------------------------------------------------- lab 3 --
export function lab3Cpu(len, blocks, tpb) {
  const names = ['A', 'B', 'C1', 'C2', 'D1', 'D2'];
  const parts = names.map(() => new Float64Array(blocks));
  const cache = names.map(() => new Float64Array(tpb));
  for (let b = 0; b < blocks; b++) {
    for (let t = 0; t < tpb; t++) {
      const valIdx = t + b * tpb;
      if (valIdx >= len) { for (const c of cache) c[t] = 0; continue; }
      const val = valIdx + 1, res1 = 1 / val, res2 = 1 / (len - valIdx);
      cache[0][t] = res1; cache[1][t] = res2;
      if (val % 2) { cache[2][t] = cache[4][t] = 0; cache[3][t] = res1; cache[5][t] = res2; }
      else { cache[3][t] = cache[5][t] = 0; cache[2][t] = res1; cache[4][t] = res2; }
    }
    // the same tree reduction as the kernel, so the f64 rounding follows the same order
    for (let i = tpb >> 1; i; i >>= 1)
      for (let t = 0; t < i; t++) for (const c of cache) c[t] += c[t + i];
    cache.forEach((c, u) => (parts[u][b] = c[0]));
  }
  const sums = parts.map((p) => p.reduce((s, v) => s + v, 0));
  return { parts, sums };
}

// ------------------------------------------------------------------- lab 4 --
// One run of execGlobal / execConst: 16 coefficients then ARR_SIZE values, from the same rand() stream.
export function lab4Input(rand, size) {
  const c = new Float32Array(20); // 16 + the constArr[16] the loop reads past the end (0 here)
  for (let i = 0; i < 16; i++) c[i] = rand() / RAND_MAX;
  const A = new Float32Array(size);
  for (let i = 0; i < size; i++) A[i] = rand() / RAND_MAX;
  return { c, A };
}
export function lab4Cpu(A, c) {
  const f = Math.fround, B = new Float32Array(A.length);
  for (let i = 0; i < A.length; i++) {
    let b = 0;
    for (let u = 1; u <= 15; u += 2)
      b = f(b + (i !== 15 ? f(f(f(c[u] * A[i]) / i) - f(f(c[u + 1] * A[i]) * i)) : f(f(f(c[0] * A[i]) * i) + f(f(c[15] * A[i]) / i))));
    B[i] = b;
  }
  return B;
}

// ------------------------------------------------------------------- lab 5 --
export function lab5Point(i) {
  const st = curandInit(i >>> 0, 0);
  return [curandUniformDouble(st), curandUniformDouble(st)];
}
// counts what kernelOptimized (outside, reversed) or kernelUnoptimized (inside) would count
export function lab5Cpu(len, reversed) {
  let counter = 0;
  for (let i = 0; i < len; i++) {
    const [x, y] = lab5Point(i);
    const r = x ** 2 + y ** 2;
    if (reversed ? r > 1 : r <= 1) counter++;
  }
  return counter;
}

// ------------------------------------------------------------ control task --
export const tanhLab = (x) => { const p = Math.exp(x), m = Math.exp(-x); return (p - m) / (p + m); };
export function controlInput(len, seed) {
  const rand = msvcRand(seed);
  return Float64Array.from({ length: len }, () => rand() / RAND_MAX);
}
// one layer: every thread seeds curand with (unsigned long long)sum, so all outputs come out equal
export function controlLayer(sum, outLen) {
  const out = new Float64Array(outLen);
  let weight = 0;
  for (let i = 0; i < outLen; i++) {
    const [lo, hi] = seedOf(sum);
    const st = curandInit(lo, hi);
    weight = curandUniformDouble(st);
    out[i] = tanhLab(weight * sum + 1);
  }
  return { out, weight, seed: Math.trunc(sum) };
}
export function controlCpu(input, layers) {
  let arr = input;
  const steps = [];
  for (let i = 1; i < layers.length; i++) {
    let sum = 0;
    for (let k = 0; k < layers[i - 1]; k++) sum += arr[k];
    const r = controlLayer(sum, layers[i]);
    steps.push({ sum, ...r });
    arr = r.out;
  }
  return { steps, output: arr };
}
