// WebGPU port of kernel from 2-lab/kernel.cu: renders one frame of tan(sin(x) + cos(x))
// into an RGBA bitmap, like the CPUAnimBitmap the lab copies each frame into.
// THREADS is the lab's threadsPerDim = ceil(sqrt(maxThreadsPerBlock)), here taken from the
// device's maxComputeInvocationsPerWorkgroup. WGSL has no f64, so the doubles are f32.

override W: i32 = 480;
override H: i32 = 360;
override MAXX: i32 = 10;
override MAXY: i32 = 8;
override MINY: i32 = -8;
override DT: f32 = 0.1;
override THREADS: u32 = 16u;

struct Frame { ticks: i32 }

@group(0) @binding(0) var<storage, read_write> bitmap: array<u32>; // 4 bytes per pixel, RGBA
@group(0) @binding(1) var<uniform> frame: Frame;

fn func(x: f32) -> f32 {
  return tan(sin(x) + cos(x));
}

@compute @workgroup_size(THREADS, THREADS, 1)
fn drawFrame(
  @builtin(workgroup_id) blockIdx: vec3<u32>,
  @builtin(local_invocation_id) threadIdx: vec3<u32>,
) {
  let blockDim = vec2<u32>(THREADS, THREADS);
  let x = i32(threadIdx.x + blockIdx.x * blockDim.x);
  let y = i32(threadIdx.y + blockIdx.y * blockDim.y);
  if (x >= W || y >= H) {
    return; // the grid is rounded up past the bitmap edges
  }
  let offset = x + y * W;
  var lesserThanOne = false;
  for (var xDelta = -0.5; xDelta <= 0.5; xDelta += 0.5) {
    let cX = (f32(x) + xDelta) / f32(W) * (f32(MAXX) + f32(frame.ticks) * DT);
    let c = abs(f32(H / (MAXY - MINY)) * (func(cX) - f32(MINY)) - f32(y));
    lesserThanOne = lesserThanOne || c <= 1.0;
  }
  // ptr[offset * 4 + 0..3] = { 255, lesserThanOne ? 0 : 255, same, 255 }, little-endian
  bitmap[offset] = select(0xffffffffu, 0xff0000ffu, lesserThanOne);
}
