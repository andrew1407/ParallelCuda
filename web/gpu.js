// WebGPU plumbing for the ported kernels: the device, WGSL loading (with //!include),
// cached compute pipelines, buffers, dispatches and read-back.

export const gpu = { device: null, info: null, reason: '', limits: null };

export async function initGpu() {
  if (!navigator.gpu) { gpu.reason = 'this browser has no WebGPU (navigator.gpu is missing)'; return false; }
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) { gpu.reason = 'no WebGPU adapter is available'; return false; }
    const a = adapter.limits;
    // ask for as much as the labs use, up to what the adapter has: 1024 threads per block
    // (labs 3 and 4) and 24 KB of shared memory (lab 3's six 1024-element caches)
    const want = {
      maxComputeInvocationsPerWorkgroup: Math.min(1024, a.maxComputeInvocationsPerWorkgroup),
      maxComputeWorkgroupSizeX: Math.min(1024, a.maxComputeWorkgroupSizeX),
      maxComputeWorkgroupSizeY: Math.min(1024, a.maxComputeWorkgroupSizeY),
      maxComputeWorkgroupStorageSize: Math.min(32768, a.maxComputeWorkgroupStorageSize),
      maxStorageBuffersPerShaderStage: Math.min(8, a.maxStorageBuffersPerShaderStage),
    };
    const device = await adapter.requestDevice({ requiredLimits: want });
    device.lost.then((e) => { gpu.device = null; gpu.reason = `the GPU device was lost (${e.message || e.reason})`; });
    gpu.device = device;
    gpu.limits = device.limits;
    const info = adapter.info || {};
    gpu.info = [info.vendor, info.architecture, info.description].filter(Boolean).join(' · ') || 'unknown adapter';
    return true;
  } catch (e) {
    gpu.reason = `WebGPU failed to start (${e.message || e})`;
    return false;
  }
}

// ------------------------------------------------------------------- shaders --
const sources = new Map();
export async function wgslSource(name) {
  if (!sources.has(name)) {
    sources.set(name, (async () => {
      const res = await fetch(`kernels/${name}`);
      if (!res.ok) throw new Error(`kernels/${name}: HTTP ${res.status}`);
      let text = await res.text();
      for (const m of [...text.matchAll(/^\/\/!include (\S+)\s*$/gm)]) text = text.replace(m[0], (await wgslSource(m[1])).trim());
      return text;
    })());
  }
  return sources.get(name);
}

const modules = new Map(), pipelines = new Map();
async function shaderModule(name) {
  if (!modules.has(name)) {
    modules.set(name, (async () => {
      const code = await wgslSource(name);
      const module = gpu.device.createShaderModule({ label: name, code });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === 'error');
      if (errors.length) throw new Error(`${name}: ${errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('; ')}`);
      return module;
    })());
  }
  return modules.get(name);
}

/** A compute pipeline for file/entry with override constants; cached per combination. */
export async function pipeline(file, entryPoint, constants = {}) {
  const key = `${file}|${entryPoint}|${JSON.stringify(constants)}`;
  if (!pipelines.has(key)) {
    const p = (async () => {
      const module = await shaderModule(file);
      gpu.device.pushErrorScope('validation');
      const pipe = await gpu.device.createComputePipelineAsync({ label: key, layout: 'auto', compute: { module, entryPoint, constants } });
      const err = await gpu.device.popErrorScope();
      if (err) throw new Error(err.message);
      return pipe;
    })();
    pipelines.set(key, p);
    p.catch(() => pipelines.delete(key));
  }
  return pipelines.get(key);
}

// ------------------------------------------------------------------- buffers --
const U = () => GPUBufferUsage;
export function storage(dataOrBytes, extra = 0) {
  const bytes = typeof dataOrBytes === 'number' ? dataOrBytes : dataOrBytes.byteLength;
  const buf = gpu.device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 4) * 4), usage: U().STORAGE | U().COPY_SRC | U().COPY_DST | extra });
  if (typeof dataOrBytes !== 'number') gpu.device.queue.writeBuffer(buf, 0, dataOrBytes);
  return buf;
}
export function uniform(data) {
  const buf = gpu.device.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 16) * 16), usage: U().UNIFORM | U().COPY_DST });
  gpu.device.queue.writeBuffer(buf, 0, data);
  return buf;
}
export function bindGroup(pipe, buffers) {
  return gpu.device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: Object.entries(buffers).map(([binding, buffer]) => ({ binding: +binding, resource: { buffer } })),
  });
}

/** Encode dispatches: jobs = [{ pipe, group, grid: [x, y, z], times }]; then copies, then submit. */
export function submit(jobs, copies = []) {
  const enc = gpu.device.createCommandEncoder();
  const pass = enc.beginComputePass();
  for (const { pipe, group, grid, times = 1 } of jobs) {
    pass.setPipeline(pipe);
    pass.setBindGroup(0, group);
    for (let t = 0; t < times; t++) pass.dispatchWorkgroups(...grid);
  }
  pass.end();
  for (const [src, dst, size] of copies) enc.copyBufferToBuffer(src, 0, dst, 0, size);
  gpu.device.queue.submit([enc.finish()]);
}
export const done = () => gpu.device.queue.onSubmittedWorkDone();

export function readback(size) {
  return gpu.device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage: U().MAP_READ | U().COPY_DST });
}
/** cudaMemcpy(DeviceToHost): copy src into a mappable buffer and return its bytes. */
export async function download(src, size, Type = Float32Array) {
  const rb = readback(size);
  const enc = gpu.device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, rb, 0, Math.ceil(size / 4) * 4);
  gpu.device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = new Type(rb.getMappedRange().slice(0, size));
  rb.unmap(); rb.destroy();
  return out;
}
export const free = (...bufs) => bufs.forEach((b) => b && b.destroy());
