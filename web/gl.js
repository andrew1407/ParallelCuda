// WebGL2 plumbing for the fragment-shader ports (kernels/*.glsl): the context, GLSL loading
// (with //!include), cached programs, textures as buffers, full-screen passes and async read-back
// through a pixel-pack buffer and a fence, so the page never waits on the GPU.

export const gl = { c: null, float: false, info: '', reason: '', maxTex: 0 };

export function initGl() {
  try {
    const canvas = typeof OffscreenCanvas === 'function' ? new OffscreenCanvas(1, 1) : document.createElement('canvas');
    const c = canvas.getContext('webgl2', { antialias: false, depth: false, stencil: false, alpha: true, powerPreference: 'high-performance' });
    if (!c) { gl.reason = 'this browser has no WebGL2'; return false; }
    // float render targets (labs 3, 4 and the control task); the integer and RGBA8 ones are core WebGL2
    gl.float = !!c.getExtension('EXT_color_buffer_float');
    gl.parallel = c.getExtension('KHR_parallel_shader_compile');
    gl.maxTex = c.getParameter(c.MAX_TEXTURE_SIZE);
    let renderer = String(c.getParameter(c.RENDERER) || 'unknown renderer');
    if (/^WebKit/.test(renderer)) { // Chrome and Safari mask it; the unmasked name is still offered there
      const ext = c.getExtension('WEBGL_debug_renderer_info');
      if (ext) renderer = String(c.getParameter(ext.UNMASKED_RENDERER_WEBGL) || renderer);
    }
    gl.info = renderer;
    canvas.addEventListener?.('webglcontextlost', (e) => { e.preventDefault(); gl.c = null; gl.reason = 'the WebGL2 context was lost'; });
    // a vertex array with no attributes: the vertex shader builds a full-screen triangle from gl_VertexID
    gl.vao = c.createVertexArray();
    gl.c = c;
    return true;
  } catch (e) {
    gl.reason = `WebGL2 failed to start (${e.message || e})`;
    return false;
  }
}

// ------------------------------------------------------------------- shaders --
const sources = new Map();
export async function glslSource(name) {
  if (!sources.has(name)) {
    sources.set(name, (async () => {
      const res = await fetch(`kernels/${name}`);
      if (!res.ok) throw new Error(`kernels/${name}: HTTP ${res.status}`);
      let text = await res.text();
      for (const m of [...text.matchAll(/^\/\/!include (\S+)\s*$/gm)]) text = text.replace(m[0], (await glslSource(m[1])).trim());
      return text;
    })());
  }
  return sources.get(name);
}

const VERTEX = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;
const HEADER = '#version 300 es\nprecision highp float;\nprecision highp int;\nprecision highp sampler2D;\nprecision highp isampler2D;\nprecision highp usampler2D;\n';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const programs = new Map();
/** A program for file with #defines; cached per combination. Compiles in parallel where the browser can. */
export function program(file, defines = {}) {
  const key = `${file}|${JSON.stringify(defines)}`;
  if (!programs.has(key)) {
    const p = (async () => {
      const c = gl.c;
      const src = HEADER + Object.entries(defines).map(([k, v]) => `#define ${k} ${v}\n`).join('') + (await glslSource(file));
      const shader = (type, text) => { const s = c.createShader(type); c.shaderSource(s, text); c.compileShader(s); return s; };
      const vs = shader(c.VERTEX_SHADER, VERTEX), fs = shader(c.FRAGMENT_SHADER, src);
      const prog = c.createProgram();
      c.attachShader(prog, vs); c.attachShader(prog, fs); c.linkProgram(prog);
      if (gl.parallel) while (!c.getProgramParameter(prog, gl.parallel.COMPLETION_STATUS_KHR)) await sleep(4);
      if (!c.getProgramParameter(prog, c.LINK_STATUS)) {
        const log = c.getShaderInfoLog(fs) || c.getShaderInfoLog(vs) || c.getProgramInfoLog(prog);
        throw new Error(`${file}: ${log}`);
      }
      c.deleteShader(vs); c.deleteShader(fs);
      return { prog, loc: new Map() };
    })();
    programs.set(key, p);
    p.catch(() => programs.delete(key));
  }
  return programs.get(key);
}

// ------------------------------------------------------------------ textures --
// a texture plays a CUDA buffer: one texel per element, NEAREST so texelFetch reads it exactly
const FMT = {
  r32i: ['R32I', 'RED_INTEGER', 'INT', Int32Array, 'RGBA_INTEGER'],
  r32ui: ['R32UI', 'RED_INTEGER', 'UNSIGNED_INT', Uint32Array, 'RGBA_INTEGER'],
  r32f: ['R32F', 'RED', 'FLOAT', Float32Array, 'RGBA'],
  rgba32f: ['RGBA32F', 'RGBA', 'FLOAT', Float32Array, 'RGBA'],
  rgba8: ['RGBA8', 'RGBA', 'UNSIGNED_BYTE', Uint8Array, 'RGBA'],
};
export function texture(w, h, fmt, data = null) {
  const c = gl.c, [internal, format, type] = FMT[fmt];
  const t = c.createTexture();
  c.bindTexture(c.TEXTURE_2D, t);
  for (const [k, v] of [[c.TEXTURE_MIN_FILTER, c.NEAREST], [c.TEXTURE_MAG_FILTER, c.NEAREST], [c.TEXTURE_WRAP_S, c.CLAMP_TO_EDGE], [c.TEXTURE_WRAP_T, c.CLAMP_TO_EDGE]]) c.texParameteri(c.TEXTURE_2D, k, v);
  c.pixelStorei(c.UNPACK_ALIGNMENT, 1);
  c.texImage2D(c.TEXTURE_2D, 0, c[internal], w, h, 0, c[format], c[type], data);
  Object.assign(t, { w, h, fmt });
  return t;
}
/** A framebuffer drawing into these textures (several = multiple render targets). */
export function target(...texs) {
  const c = gl.c, fb = c.createFramebuffer();
  c.bindFramebuffer(c.FRAMEBUFFER, fb);
  texs.forEach((t, i) => c.framebufferTexture2D(c.FRAMEBUFFER, c.COLOR_ATTACHMENT0 + i, c.TEXTURE_2D, t, 0));
  c.drawBuffers(texs.map((_, i) => c.COLOR_ATTACHMENT0 + i));
  const status = c.checkFramebufferStatus(c.FRAMEBUFFER);
  if (status !== c.FRAMEBUFFER_COMPLETE) throw new Error(`WebGL2 can't render into ${texs[0].fmt} textures here (framebuffer status 0x${status.toString(16)})`);
  fb.texs = texs;
  return fb;
}

/** One kernel launch: a full-screen pass over w×h fragments of fb. uni: { name: ['1i', v, …] }, tex: { name: texture }. */
export function draw({ prog, loc }, fb, w, h, { uni = {}, tex = {} } = {}) {
  const c = gl.c;
  c.useProgram(prog);
  const at = (name) => { if (!loc.has(name)) loc.set(name, c.getUniformLocation(prog, name)); return loc.get(name); };
  Object.entries(tex).forEach(([name, t], unit) => {
    c.activeTexture(c.TEXTURE0 + unit);
    c.bindTexture(c.TEXTURE_2D, t);
    c.uniform1i(at(name), unit);
  });
  for (const [name, [fn, ...v]] of Object.entries(uni)) c[`uniform${fn}`](at(name), ...v);
  c.bindFramebuffer(c.FRAMEBUFFER, fb);
  c.viewport(0, 0, w, h);
  c.bindVertexArray(gl.vao);
  c.drawArrays(c.TRIANGLES, 0, 3);
}

// waits for a fence without blocking: polls it from timers until the GPU has passed it
async function fence() {
  const c = gl.c, sync = c.fenceSync(c.SYNC_GPU_COMMANDS_COMPLETE, 0);
  c.flush();
  try {
    for (;;) {
      const s = c.clientWaitSync(sync, 0, 0);
      if (s === c.ALREADY_SIGNALED || s === c.CONDITION_SATISFIED) return;
      if (s === c.WAIT_FAILED) throw new Error('WebGL2 fence wait failed');
      await sleep(1);
    }
  } finally { c.deleteSync(sync); }
}
/** cudaDeviceSynchronize: resolves when every pass so far has finished. */
export const finish = () => fence();

/** cudaMemcpy(DeviceToHost) of attachment `at` of fb, w×h texels: one value per texel (four for rgba formats). */
export async function read(fb, w, h, at = 0) {
  const c = gl.c, fmt = fb.texs[at].fmt, [, , type, Arr, readFormat] = FMT[fmt];
  const n = w * h * 4, pbo = c.createBuffer();
  c.bindFramebuffer(c.READ_FRAMEBUFFER, fb);
  c.readBuffer(c.COLOR_ATTACHMENT0 + at);
  c.bindBuffer(c.PIXEL_PACK_BUFFER, pbo);
  c.bufferData(c.PIXEL_PACK_BUFFER, n * Arr.BYTES_PER_ELEMENT, c.STREAM_READ);
  c.readPixels(0, 0, w, h, c[readFormat], c[type], 0);
  c.bindBuffer(c.PIXEL_PACK_BUFFER, null);
  await fence();
  const all = new Arr(n);
  c.bindBuffer(c.PIXEL_PACK_BUFFER, pbo);
  c.getBufferSubData(c.PIXEL_PACK_BUFFER, 0, all);
  c.bindBuffer(c.PIXEL_PACK_BUFFER, null);
  c.deleteBuffer(pbo);
  if (fmt.startsWith('rgba')) return all;
  const one = new Arr(w * h);
  for (let i = 0; i < one.length; i++) one[i] = all[i * 4];
  return one;
}

export function release(...objs) {
  const c = gl.c;
  if (!c) return;
  for (const o of objs) {
    if (!o) continue;
    if (o instanceof WebGLFramebuffer) c.deleteFramebuffer(o); else c.deleteTexture(o);
  }
}

/** A 2D layout for n elements: rows of up to 2048 texels. */
export const layout = (n) => { const w = Math.max(1, Math.min(n, 2048)); return [w, Math.max(1, Math.ceil(n / w))]; };
