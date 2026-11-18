// ParallelCuda showcase: each lab's CUDA kernel ported to a WebGPU compute shader (kernels/*.wgsl),
// checked and timed against a JavaScript CPU reference (cpu.js). Without WebGPU the labs run
// the CPU reference only.

import * as cpu from './cpu.js';
import { gpu, initGpu, wgslSource, pipeline, storage, uniform, bindGroup, submit, done, readback, download, free } from './gpu.js';

const REPO = 'https://github.com/Andrew1407/ParallelCuda';
const SRC = `${REPO}/blob/main`;
const $ = (s, root = document) => root.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmt = (v, d = 4) => {
  if (typeof v !== 'number' || !Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e6 || a < 1e-3)) return v.toExponential(2);
  return String(+v.toFixed(d));
};
const msFmt = (v) => (v === null || v === undefined ? '—' : `${fmt(v, v < 10 ? 3 : 1)} ms`);
const range = (n, f = (i) => i) => Array.from({ length: n }, (_, i) => f(i));
const pause = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
const stat = (value, label, cls = '') => `<div class="stat ${cls}"><b>${value}</b><span>${label}</span></div>`;
const now = () => performance.now();
const mean = (xs) => Array.from(xs).reduce((s, v) => s + v, 0) / xs.length;
const median = (xs) => { const s = [...xs].sort((a, b) => a - b); return s.length ? (s[(s.length - 1) >> 1] + s[s.length >> 1]) / 2 : NaN; };
const speedup = (g, c) => (g && c ? `${fmt(c / g, 2)}×` : '—');
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const rgb = (hex) => { const h = hex.replace('#', ''); return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)); };
// printf("%f") and friends
const fixed = (v, d) => (Number.isFinite(v) ? v.toFixed(d) : v > 0 ? 'inf' : v < 0 ? '-inf' : 'nan');
const coutNum = (v) => (Number.isFinite(v) ? String(+v.toPrecision(6)) : v > 0 ? 'inf' : 'nan');

let gpuOk = false;
const gpuReady = initGpu().then((ok) => { gpuOk = ok; showGpuStatus(); return ok; });
const maxThreads = () => (gpuOk ? gpu.limits.maxComputeInvocationsPerWorkgroup : 0);

// ------------------------------------------------------------------ charts --
const charts = new Map();
let chartSeq = 0;

function niceStep(span, count) {
  const raw = span / Math.max(1, count);
  const p = 10 ** Math.floor(Math.log10(raw || 1));
  const m = raw / p;
  return (m >= 5 ? 10 : m >= 2 ? 5 : m >= 1 ? 2 : 1) * p;
}
function ticks(min, max, count) {
  if (min === max) { min -= Math.abs(min) * 0.1 || 1; max += Math.abs(max) * 0.1 || 1; }
  const step = niceStep(max - min, count);
  const lo = Math.floor(min / step) * step, hi = Math.ceil(max / step) * step;
  const out = [];
  for (let v = lo; v <= hi + step / 2; v += step) out.push(+v.toPrecision(12));
  return { lo, hi, ticks: out };
}

/** spec: { title, x: label, y: label, series: [{ name, xs, ys, color }], height, dots } */
function chart(spec) {
  const id = `c${++chartSeq}`;
  charts.set(id, spec);
  const keys = spec.series.length > 1
    ? `<span class="keys">${spec.series.map((s) => `<span><i style="background:var(${s.color})"></i>${esc(s.name)}</span>`).join('')}</span>` : '';
  return `<figure class="chart" data-chart="${id}" data-copyable data-title="${esc(spec.title)}"><figcaption><span>${esc(spec.title)}</span>${keys}${actions('png', 'csv')}</figcaption><div class="plot"></div></figure>`;
}

function drawCharts(root = document) {
  root.querySelectorAll('figure.chart').forEach((fig) => {
    const spec = charts.get(fig.dataset.chart);
    const plot = $('.plot', fig);
    if (!spec || !plot.offsetParent) return;
    drawChart(plot, spec);
  });
}
let resizeTimer;
window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => drawCharts(), 120); });

function drawChart(plot, spec) {
  const W = Math.max(260, plot.clientWidth), H = spec.height || 230;
  const M = { l: 58, r: 14, t: 8, b: 38 };
  const all = spec.series.flatMap((s) => s.xs.map((x, i) => [x, s.ys[i]])).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  if (!all.length) { plot.innerHTML = '<p class="hint">No data.</p>'; return; }
  const xr = ticks(Math.min(...all.map((p) => p[0])), Math.max(...all.map((p) => p[0])), Math.max(3, Math.floor((W - M.l) / 90)));
  const yr = ticks(Math.min(...all.map((p) => p[1])), Math.max(...all.map((p) => p[1])), 5);
  const x0 = spec.tightX === false ? xr.lo : Math.min(...all.map((p) => p[0])), x1 = spec.tightX === false ? xr.hi : Math.max(...all.map((p) => p[0]));
  const X = (v) => M.l + ((v - x0) / (x1 - x0 || 1)) * (W - M.l - M.r);
  const Y = (v) => H - M.b - ((v - yr.lo) / (yr.hi - yr.lo || 1)) * (H - M.t - M.b);
  const xt = xr.ticks.filter((v) => v >= x0 - 1e-9 && v <= x1 + 1e-9);
  const grid = yr.ticks.map((v) => `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${M.l - 8}" y="${Y(v)}" text-anchor="end" dominant-baseline="middle">${fmt(v, 6)}</text>`).join('')
    + xt.map((v) => `<line x1="${X(v)}" x2="${X(v)}" y1="${H - M.b}" y2="${H - M.b + 4}" stroke="var(--muted)"/><text x="${X(v)}" y="${H - M.b + 16}" text-anchor="middle">${fmt(v, 6)}</text>`).join('');
  const axes = `<line x1="${M.l}" x2="${W - M.r}" y1="${H - M.b}" y2="${H - M.b}" stroke="var(--line)"/>`
    + (yr.lo < 0 && yr.hi > 0 ? `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--muted)" stroke-opacity=".45"/>` : '')
    + `<text x="${(M.l + W - M.r) / 2}" y="${H - 4}" text-anchor="middle">${esc(spec.x)}</text>`
    + `<text transform="translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">${esc(spec.y)}</text>`;
  const lines = spec.series.map((s) => {
    const ok = (i) => Number.isFinite(s.ys[i]);
    const pts = s.xs.map((x, i) => (ok(i) ? `${X(x).toFixed(1)},${Y(s.ys[i]).toFixed(1)}` : '')).filter(Boolean).join(' ');
    const dots = spec.dots || s.xs.length <= 30 ? s.xs.map((x, i) => (ok(i) ? `<circle cx="${X(x)}" cy="${Y(s.ys[i])}" r="2.6" fill="var(${s.color})"/>` : '')).join('') : '';
    return `<polyline points="${pts}" fill="none" stroke="var(${s.color})" stroke-width="${s.width || 1.8}" stroke-linejoin="round" ${s.dash ? 'stroke-dasharray="5 4"' : ''}/>${dots}`;
  }).join('');
  plot.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="${esc(spec.title)}">${grid}${axes}${lines}<line class="cross" y1="${M.t}" y2="${H - M.b}" stroke="var(--muted)" stroke-dasharray="3 3" visibility="hidden"/><g class="marks"></g></svg><div class="tip" hidden></div>`;

  const svg = $('svg', plot), tip = $('.tip', plot), cross = $('.cross', svg), marks = $('.marks', svg);
  const nearest = (xs, v) => {
    let lo = 0, hi = xs.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] < v) lo = m; else hi = m; }
    return Math.abs(xs[lo] - v) <= Math.abs(xs[hi] - v) ? lo : hi;
  };
  const move = (e) => {
    const rect = svg.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    if (px < M.l - 6 || px > W - M.r + 6) return hide();
    const v = x0 + ((px - M.l) / (W - M.l - M.r)) * (x1 - x0);
    const hits = spec.series.map((s) => { const i = nearest(s.xs, v); return { s, x: s.xs[i], y: s.ys[i] }; });
    const hx = X(hits[0].x);
    cross.setAttribute('x1', hx); cross.setAttribute('x2', hx); cross.setAttribute('visibility', 'visible');
    marks.innerHTML = hits.filter((h) => Number.isFinite(h.y)).map((h) => `<circle cx="${X(h.x)}" cy="${Y(h.y)}" r="4" fill="var(--panel)" stroke="var(${h.s.color})" stroke-width="2"/>`).join('');
    tip.innerHTML = `<div class="tx">${esc(spec.x)} = ${fmt(hits[0].x, 6)}</div>` + hits.map((h) => `<div><i style="background:var(${h.s.color})"></i>${esc(h.s.name || spec.y)}: ${fmt(h.y, 8)}</div>`).join('');
    tip.hidden = false;
    const left = (hx / W) * rect.width;
    const tw = tip.offsetWidth;
    tip.style.left = `${left + 12 + tw > rect.width ? Math.max(0, left - tw - 12) : left + 12}px`;
    tip.style.top = `${M.t}px`;
  };
  const hide = () => { tip.hidden = true; cross.setAttribute('visibility', 'hidden'); marks.innerHTML = ''; };
  svg.addEventListener('pointermove', move);
  svg.addEventListener('pointerdown', move);
  svg.addEventListener('pointerleave', hide);
}

// ---------------------------------------------------------- GPU / CPU bits --
const engineNote = () => (gpuOk ? '' : '<p class="note">WebGPU isn\'t available, so this ran the JavaScript CPU reference only.</p>');
const verdict = (ok, good, bad) => stat(ok ? good : bad, 'verification', ok ? 'good' : 'bad');
const timeStats = (g, c) => `${stat(gpuOk ? msFmt(g) : 'n/a', 'GPU (WebGPU)')}${stat(msFmt(c), 'CPU (JavaScript)')}${gpuOk ? stat(speedup(g, c), 'GPU speed-up') : ''}`;
const launch = (text) => stat(`<code>${esc(text)}</code>`, 'launch', 'launch');

// ------------------------------------------------------------------- lab 1 --
async function runLab1(p) {
  const N = p.size, b = 1;
  const { A, B } = cpu.lab1Input(N, p.seed);
  await pause();
  let t = now();
  const Ccpu = cpu.lab1Cpu(A, B, N);
  const cpuMs = now() - t;
  let Cgpu = null, gpuMs = null;
  if (await gpuReady) {
    const pipe = await pipeline('lab1.wgsl', 'matrixMult', { MATRIX_SIZE: N });
    const Ad = storage(A), Bd = storage(B), Cd = storage(A.byteLength);
    const group = bindGroup(pipe, { 0: Ad, 1: Bd, 2: Cd });
    const grid = [N, N, 1]; // <<<dim3(MATRIX_SIZE, MATRIX_SIZE), 1>>>
    submit([{ pipe, group, grid: [1, 1, 1] }]); await done(); // warm-up: first use of the pipeline
    t = now();
    submit([{ pipe, group, grid }]);
    await done();
    gpuMs = now() - t;
    Cgpu = await download(Cd, A.byteLength, Int32Array);
    free(Ad, Bd, Cd);
  }
  const C = Cgpu || Ccpu;
  let bad = null;
  if (Cgpu) for (let i = 0; i < N * N && !bad; i++) if (Cgpu[i] !== Ccpu[i]) bad = { i: Math.floor(i / N), j: i % N, sum: Ccpu[i], v: Cgpu[i] };
  return { N, A, B, C, Ccpu, gpuMs, cpuMs, bad };
}

const matrices = new Map();
let matrixSeq = 0;
function renderLab1(d, p) {
  const conf = `<<<dim3(${d.N}, ${d.N}), 1>>>`;
  const console_ = d.bad
    ? `Result verification failed at element [${d.bad.i}, ${d.bad.j}]!\nsum = ${d.bad.sum}, matrixCh[i * MATRIX_SIZE + j] = ${d.bad.v}`
    : `Calculated: ${fixed((gpuOk ? d.gpuMs : d.cpuMs) * 1e-3, 6)} seconds.`;
  const id = `m${++matrixSeq}`;
  matrices.set(id, { A: d.A, B: d.B, C: d.C, N: d.N, seed: p.seed });
  return `${engineNote()}
    <div class="stats">${timeStats(d.gpuMs, d.cpuMs)}${gpuOk ? verdict(!d.bad, 'C = A × B', 'mismatch') : ''}${launch(conf)}</div>
    <div class="two-col">
      <div data-copyable data-title="C = A × B, ${d.N}×${d.N}"><div class="copy-head"><h2>Result heatmap <small>· C, ${d.N}×${d.N}</small></h2>${actions('png')}</div><canvas class="heat" data-heat="${id}"></canvas><div class="legend"><span>low</span><span class="ramp"></span><span>high</span></div></div>
      <div data-copyable data-matrix="${id}" data-which="C" data-title="Matrix preview" style="min-width:0">
        <div class="copy-head"><h2>Preview <small>· top-left corner</small></h2>${actions('table', 'tsv', 'download')}</div>
        <div class="tabs">${['A', 'B', 'C'].map((m) => `<button type="button" class="tab" data-pick="${m}" aria-selected="${m === 'C'}">${m}</button>`).join('')}</div>
        <div class="table-wrap matrix-preview"></div>
        <p class="hint">Copy TSV and Download take the whole selected matrix (${d.N}×${d.N}).</p>
      </div>
    </div>
    ${consoleBox(console_)}`;
}
function previewMatrix(box) {
  const m = matrices.get(box.dataset.matrix), which = box.dataset.which, M = m[which], n = Math.min(10, m.N);
  box.querySelectorAll('.tab').forEach((t) => t.setAttribute('aria-selected', t.dataset.pick === which));
  $('.matrix-preview', box).innerHTML = `<table><thead><tr><th>${which}</th>${range(n, (j) => `<th class="num">${j}</th>`).join('')}<th>…</th></tr></thead><tbody>${range(n, (i) => `<tr><th>${i}</th>${range(n, (j) => `<td class="num">${M[i * m.N + j]}</td>`).join('')}<td class="muted">…</td></tr>`).join('')}</tbody></table>`;
}
function drawHeat(canvas) {
  const m = matrices.get(canvas.dataset.heat);
  if (!m) return;
  const N = m.N, C = m.C, k = Math.max(1, Math.ceil(420 / N));
  let lo = Infinity, hi = -Infinity;
  for (const v of C) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const a = rgb(cssVar('--panel')), b = rgb(cssVar('--accent'));
  const img = new ImageData(N, N);
  for (let i = 0; i < N * N; i++) {
    const t = hi > lo ? (C[i] - lo) / (hi - lo) : 0.5;
    for (let c = 0; c < 3; c++) img.data[i * 4 + c] = a[c] + (b[c] - a[c]) * t;
    img.data[i * 4 + 3] = 255;
  }
  const tmp = document.createElement('canvas');
  tmp.width = tmp.height = N;
  tmp.getContext('2d').putImageData(img, 0, 0);
  canvas.width = canvas.height = N * k;
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tmp, 0, 0, N * k, N * k);
}
const matrixTsv = (m, which) => range(m.N, (i) => Array.from(m[which].subarray(i * m.N, (i + 1) * m.N)).join('\t')).join('\n');

// ------------------------------------------------------------------- lab 2 --
const anim = { playing: false, engine: 'gpu', ticks: 1, raf: 0, ctx: null, busy: false, frames: 0, fpsT: 0, fps: 0, last: 0 };

async function lab2Gpu(p) {
  const threadsPerDim = Math.ceil(Math.sqrt(maxThreads()));
  const t = threadsPerDim ** 2 > maxThreads() ? Math.floor(Math.sqrt(maxThreads())) : threadsPerDim;
  const blocks = [Math.ceil(p.W / t), Math.ceil(p.H / t), 1];
  const pipe = await pipeline('lab2.wgsl', 'drawFrame', { W: p.W, H: p.H, MAXX: p.MAXX, MAXY: p.MAXY, MINY: p.MINY, DT: p.DT, THREADS: t });
  const size = p.W * p.H * 4;
  const bitmap = storage(size), frame = uniform(new Int32Array(4)), rb = readback(size);
  const group = bindGroup(pipe, { 0: bitmap, 1: frame });
  return {
    threads: t, blocks,
    async render(ticks, px) {
      gpu.device.queue.writeBuffer(frame, 0, new Int32Array([ticks, 0, 0, 0]));
      submit([{ pipe, group, grid: blocks }], [[bitmap, rb, size]]);
      await rb.mapAsync(GPUMapMode.READ);
      px.set(new Uint32Array(rb.getMappedRange()));
      rb.unmap();
    },
    free: () => free(bitmap, frame, rb),
  };
}

async function runLab2(p) {
  stopAnim();
  anim.ctx?.gpu?.free();
  const px = new Uint32Array(p.W * p.H), pxCpu = new Uint32Array(p.W * p.H);
  const frames = p.frames;
  await pause();
  let t = now();
  for (let k = 1; k <= frames; k++) cpu.lab2Frame(pxCpu, k, p);
  const cpuMs = (now() - t) / frames;
  let g = null, gpuMs = null, mismatch = null;
  if (await gpuReady) {
    g = await lab2Gpu(p);
    await g.render(1, px); // warm-up
    t = now();
    for (let k = 1; k <= frames; k++) await g.render(k, px);
    gpuMs = (now() - t) / frames;
    mismatch = 0;
    for (let i = 0; i < px.length; i++) mismatch += px[i] !== pxCpu[i];
  }
  anim.ctx = { p, gpu: g, px, cpuMs, gpuMs };
  anim.engine = g ? 'gpu' : 'cpu';
  anim.ticks = frames;
  return { p, frames, cpuMs, gpuMs, mismatch, threads: g?.threads, blocks: g?.blocks, px: g ? px : pxCpu };
}

function renderLab2(d) {
  const conf = d.threads ? `<<<dim3(${d.blocks[0]}, ${d.blocks[1]}), dim3(${d.threads}, ${d.threads})>>>` : '—';
  return `${engineNote()}
    <div class="stats">${stat(gpuOk ? msFmt(d.gpuMs) : 'n/a', 'GPU per frame')}${stat(msFmt(d.cpuMs), 'CPU per frame')}${gpuOk ? stat(speedup(d.gpuMs, d.cpuMs), 'GPU speed-up') : ''}${gpuOk ? stat(d.mismatch, `pixels differing in frame ${d.frames}`, d.mismatch / d.px.length < 0.01 ? 'good' : 'bad') : ''}${launch(conf)}</div>
    <div data-copyable data-title="tan(sin(x) + cos(x)), frame" class="anim-box">
      <div class="copy-head"><h2>Animation <small>· <span class="ticks"></span></small></h2>${actions('png')}</div>
      <div class="anim-controls">
        <button type="button" class="mini play">Pause</button>
        <button type="button" class="mini step">Step</button>
        <button type="button" class="mini restart">Restart</button>
        <div class="seg" role="radiogroup" aria-label="Renderer">${[['gpu', 'GPU'], ['cpu', 'CPU']].map(([v, t]) => `<label><input type="radio" name="engine" value="${v}"${v === anim.engine ? ' checked' : ''}${v === 'gpu' && !gpuOk ? ' disabled' : ''}>${t}</label>`).join('')}</div>
        <span class="muted fps"></span>
      </div>
      <canvas class="bitmap" width="${d.p.W}" height="${d.p.H}"></canvas>
    </div>`;
}

function blit(canvas, px, W, H) {
  const img = new ImageData(W, H), dst = new Uint32Array(img.data.buffer);
  // glDrawPixels puts row 0 at the bottom of the window
  for (let y = 0; y < H; y++) dst.set(px.subarray(y * W, (y + 1) * W), (H - 1 - y) * W);
  canvas.getContext('2d').putImageData(img, 0, 0);
}
async function drawTick(box) {
  const c = anim.ctx, p = c.p;
  if (anim.busy) return;
  anim.busy = true;
  const t = now();
  try {
    if (anim.engine === 'gpu' && c.gpu) await c.gpu.render(anim.ticks, c.px);
    else cpu.lab2Frame(c.px, anim.ticks, p);
  } finally { anim.busy = false; }
  anim.last = now() - t;
  blit($('canvas', box), c.px, p.W, p.H);
  $('.ticks', box).textContent = `ticks = ${anim.ticks}, x from 0 to ${fmt(p.MAXX + anim.ticks * p.DT, 2)}`;
  anim.frames++;
  if (now() - anim.fpsT > 500) { anim.fps = (anim.frames * 1000) / (now() - anim.fpsT); anim.frames = 0; anim.fpsT = now(); }
  $('.fps', box).textContent = `${anim.engine.toUpperCase()} · ${fmt(anim.last, 2)} ms per frame · ${Math.round(anim.fps)} fps`;
}
function loop(box) {
  if (!anim.playing) return;
  drawTick(box).then(() => { anim.ticks++; anim.raf = requestAnimationFrame(() => loop(box)); });
}
function startAnim(box) { if (anim.playing || !anim.ctx || !box.isConnected) return; anim.playing = true; anim.fpsT = now(); anim.frames = 0; $('.play', box).textContent = 'Pause'; loop(box); }
function stopAnim() { anim.playing = false; cancelAnimationFrame(anim.raf); document.querySelectorAll('.anim-box .play').forEach((b) => (b.textContent = 'Play')); }
function bindLab2(out) {
  const box = $('.anim-box', out);
  $('.play', box).onclick = () => (anim.playing ? stopAnim() : startAnim(box));
  $('.step', box).onclick = () => { stopAnim(); anim.ticks++; drawTick(box); };
  $('.restart', box).onclick = () => { anim.ticks = 1; if (!anim.playing) drawTick(box); };
  box.querySelectorAll('input[name=engine]').forEach((r) => (r.onchange = () => { anim.engine = r.value; anim.fpsT = now(); anim.frames = 0; }));
  blit($('canvas', box), anim.ctx.px, anim.ctx.p.W, anim.ctx.p.H);
  startAnim(box);
}

// ------------------------------------------------------------------- lab 3 --
const SUMS = [
  ['A', 'Σ 1/i', 'i = 1 … N'],
  ['B', 'Σ 1/(N − i)', 'the same terms in reverse'],
  ['C1', 'Σ 1/i, i even', 'C = C1 + C2'],
  ['C2', 'Σ 1/i, i odd', ''],
  ['D1', 'Σ 1/(N − i + 1), i even', 'D = D1 + D2'],
  ['D2', 'Σ 1/(N − i + 1), i odd', ''],
];
async function runLab3(p) {
  const { len, blocks } = p, tpb = Number(p.tpb);
  await pause();
  let t = now();
  const ref = cpu.lab3Cpu(len, blocks, tpb);
  const cpuMs = now() - t;
  let parts = null, gpuMs = null, wg = null, gpuSkip = '';
  if (await gpuReady) {
    wg = Math.min(tpb, maxThreads());
    const need = 6 * tpb * 4, have = gpu.limits.maxComputeWorkgroupStorageSize;
    if (need > have) gpuSkip = `The kernel needs ${need} bytes of shared memory per block and this GPU allows ${have}, so only the CPU reference ran.`;
    else {
      const pipe = await pipeline('lab3.wgsl', 'calculateSums', { THREADS_PER_BLOCK: tpb, WG: wg, ARRAY_LENGTH: len });
      const bufs = range(6, () => storage(blocks * 4));
      const group = bindGroup(pipe, Object.fromEntries(bufs.map((b, i) => [i, b])));
      submit([{ pipe, group, grid: [blocks, 1, 1] }]); await done();
      t = now();
      submit([{ pipe, group, grid: [blocks, 1, 1] }]);
      parts = await Promise.all(bufs.map((b) => download(b, blocks * 4)));
      gpuMs = now() - t;
      free(...bufs);
    }
  }
  const sums = parts ? parts.map((a) => a.reduce((s, v) => s + v, 0)) : ref.sums; // host loop in double
  return { len, blocks, tpb, wg, parts, ref, sums, gpuMs, cpuMs, gpuSkip };
}
function lab3Console(s) {
  const f = (v) => fixed(v, 30);
  return `A = ${f(s[0])}\nB = ${f(s[1])}\nC = C1 + C2 = ${f(s[2])} + ${f(s[3])}\n = ${f(s[2] + s[3])}\nD = D1 + D2 = ${f(s[4])} + ${f(s[5])}\n = ${f(s[4] + s[5])}`;
}
function renderLab3(d) {
  const ran = !!d.parts;
  const covered = Math.min(d.len, d.blocks * d.tpb);
  const rel = d.sums.map((v, i) => Math.abs(v - d.ref.sums[i]) / Math.abs(d.ref.sums[i] || 1));
  const maxRel = Math.max(...rel);
  const conf = `<<<${d.blocks}, ${d.tpb}>>>` + (d.wg && d.wg < d.tpb ? ` as ${d.blocks} × ${d.wg} invocations` : '');
  const rows = SUMS.map(([n, f, note], i) => `<tr><td><span class="sw" style="background:var(--c${i + 1})"></span><b>${n}</b></td><td>${f}</td><td class="num">${ran ? d.sums[i].toFixed(10) : '—'}</td><td class="num">${d.ref.sums[i].toFixed(15)}</td><td class="num">${ran ? fmt(rel[i], 3) : '—'}</td><td class="muted">${note}</td></tr>`).join('');
  return `${engineNote()}${d.gpuSkip ? `<p class="note">${esc(d.gpuSkip)}</p>` : ''}
    ${covered < d.len ? `<p class="note">${d.blocks} × ${d.tpb} threads cover only the first ${covered} of ${d.len} terms; the rest are left out, as in the lab.</p>` : ''}
    ${d.wg && d.wg < d.tpb ? `<p class="note">This GPU runs at most ${d.wg} invocations per workgroup, so each invocation plays ${d.tpb / d.wg} of the block's ${d.tpb} threads. The shared cache and the reduction tree stay the same.</p>` : ''}
    <div class="stats">${stat(d.sums[0].toFixed(6), 'A')}${stat(d.sums[1].toFixed(6), 'B')}${stat((d.sums[2] + d.sums[3]).toFixed(6), 'C = C1 + C2')}${stat((d.sums[4] + d.sums[5]).toFixed(6), 'D = D1 + D2')}
      ${timeStats(d.gpuMs, d.cpuMs)}${ran ? verdict(maxRel < 1e-5, `rel. error ${fmt(maxRel, 2)}`, `rel. error ${fmt(maxRel, 2)}`) : ''}${launch(conf)}</div>
    <div data-copyable><div class="copy-head"><h2>Six sums <small>· GPU in f32 (WGSL has no double), CPU in f64 like the lab</small></h2>${actions('table')}</div>
      <div class="table-wrap"><table><thead><tr><th>Sum</th><th>Terms</th><th class="num">GPU (f32)</th><th class="num">CPU (f64)</th><th class="num">rel. error</th><th></th></tr></thead><tbody>${rows}</tbody></table></div></div>
    <div class="charts" style="margin-top:16px">${chart({ title: 'Per-block partial sums (what each block writes back)', x: 'block', y: 'partial sum', series: SUMS.map(([n], i) => ({ name: n, xs: range(d.blocks), ys: Array.from((d.parts || d.ref.parts)[i]), color: `--c${i + 1}` })) })}</div>
    ${consoleBox(lab3Console(d.sums))}`;
}

// ------------------------------------------------------------------- lab 4 --
async function runLab4(p) {
  const size = p.size, K = p.repeats, R = p.runs;
  const rand = cpu.msvcRand(p.seed);
  const g = cpu.lab4Input(rand, size), c = cpu.lab4Input(rand, size); // execGlobal, then execConst
  await pause();
  let t = now();
  for (let k = 0; k < K; k++) cpu.lab4Cpu(g.A, g.c);
  const cpuMs = (now() - t) / K;
  const ref = { g: cpu.lab4Cpu(g.A, g.c), c: cpu.lab4Cpu(c.A, c.c) };
  const res = { size, K, R, g, c, ref, cpuMs, wg: null, times: { g: [], c: [] }, B: null };
  if (!(await gpuReady)) return res;
  const wg = Math.min(size, maxThreads());
  res.wg = wg;
  const grid = [Math.ceil(size / wg), 1, 1];
  const pg = await pipeline('lab4.wgsl', 'kernelGlobal', { ARR_SIZE: size, WG: wg });
  const pc = await pipeline('lab4.wgsl', 'kernelConst', { ARR_SIZE: size, WG: wg });
  // one run = what the lab brackets with cudaEvents: allocate, upload, K launches, copy back
  const once = async (kind) => {
    const isG = kind === 'g', data = isG ? g : c;
    const t0 = now();
    const Ad = storage(data.A), Bd = storage(size * 4);
    const cd = isG ? storage(data.c) : uniform(data.c);
    const pipe = isG ? pg : pc;
    const group = bindGroup(pipe, isG ? { 0: Ad, 1: Bd, 2: cd } : { 0: Ad, 1: Bd, 3: cd });
    submit([{ pipe, group, grid, times: K }]);
    const B = await download(Bd, size * 4);
    const ms = now() - t0;
    free(Ad, Bd, cd);
    return { ms, B };
  };
  await once('g'); await once('c'); // warm-up
  const B = {};
  for (let r = 0; r < R; r++) {
    for (const kind of r % 2 ? ['c', 'g'] : ['g', 'c']) {
      const o = await once(kind);
      res.times[kind].push(o.ms);
      B[kind] = o.B;
    }
  }
  res.B = B;
  return res;
}
function lab4Err(gpuB, cpuB) {
  let max = 0;
  for (let i = 0; i < cpuB.length; i++) max = Math.max(max, Math.abs(gpuB[i] - cpuB[i]) / Math.max(1, Math.abs(cpuB[i])));
  return { max };
}
function renderLab4(d) {
  const ran = !!d.B;
  const mg = median(d.times.g), mc = median(d.times.c);
  const eg = ran ? lab4Err(d.B.g, d.ref.g) : null, ec = ran ? lab4Err(d.B.c, d.ref.c) : null;
  const ok = ran && eg.max < 1e-4 && ec.max < 1e-4;
  const conf = d.wg ? `<<<${Math.ceil(d.size / d.wg)}, ${d.wg}>>> × ${d.K}` : '—';
  const n = Math.min(20, d.size);
  const val = (v) => (Number.isFinite(v) ? v.toPrecision(7) : String(v));
  const rows = range(n, (i) => `<tr><td class="num">${i}</td><td class="num">${val(d.g.A[i])}</td><td class="num">${ran ? val(d.B.g[i]) : '—'}</td><td class="num">${val(d.ref.g[i])}</td><td class="num">${val(d.c.A[i])}</td><td class="num">${ran ? val(d.B.c[i]) : '—'}</td><td class="num">${val(d.ref.c[i])}</td></tr>`).join('');
  const runs = range(d.R, (i) => i + 1);
  return `${engineNote()}
    <p class="hint" style="margin:0 0 14px">The formula divides by <code>i</code>, so the kernel returns at <code>i = 0</code> and leaves <code>B[0]</code> at 0.</p>
    <div class="stats">${stat(gpuOk ? msFmt(mg) : 'n/a', 'global (storage), median run')}${stat(gpuOk ? msFmt(mc) : 'n/a', 'constant (uniform), median run')}${gpuOk ? stat(`${fmt(mg / mc, 3)}×`, 'global / constant') : ''}${stat(msFmt(d.cpuMs), 'CPU, one pass')}
      ${ran ? verdict(ok, 'both match CPU', 'mismatch') : ''}${launch(conf)}</div>
    ${gpuOk ? `<div class="charts">${chart({ title: `Run time: ${d.K} launches incl. upload and read-back`, x: 'run', y: 'time, ms', series: [{ name: 'global memory', xs: runs, ys: d.times.g, color: '--c2' }, { name: 'constant (uniform)', xs: runs, ys: d.times.c, color: '--c1' }] })}</div>` : ''}
    <div data-copyable style="margin-top:16px"><div class="copy-head"><h2>First ${n} elements <small>· execGlobal and execConst draw their own data from one rand() stream</small></h2>${actions('table')}</div>
      <div class="table-wrap"><table><thead><tr><th class="num">i</th><th class="num">A (global)</th><th class="num">B global, GPU</th><th class="num">B global, CPU</th><th class="num">A (const)</th><th class="num">B const, GPU</th><th class="num">B const, CPU</th></tr></thead><tbody>${rows}</tbody></table></div></div>
    ${consoleBox(gpuOk ? `Global: ${coutNum(d.times.g[0])} ms.\nConstant: ${coutNum(d.times.c[0])} ms.` : `CPU: ${coutNum(d.cpuMs)} ms.`)}`;
}

// ------------------------------------------------------------------- lab 5 --
async function runLab5(p) {
  const { len, blocks, threads } = p;
  if (gpuOk && threads > maxThreads()) throw new Error(`This GPU allows at most ${maxThreads()} threads per workgroup.`);
  await pause();
  const out = {};
  for (const [key, reversed] of [['opt', true], ['unopt', false]]) {
    const t = now();
    const counter = cpu.lab5Cpu(len, reversed);
    out[key] = { cpu: counter, cpuMs: now() - t, gpu: null, gpuMs: null };
  }
  if (await gpuReady) {
    for (const [key, entry] of [['opt', 'kernelOptimized'], ['unopt', 'kernelUnoptimized']]) {
      const pipe = await pipeline('lab5.wgsl', entry, { ARR_LENGTH: len, THREADS: threads });
      const counter = storage(new Uint32Array(1));
      const group = bindGroup(pipe, { 0: counter });
      submit([{ pipe, group, grid: [blocks, 1, 1] }]); await done(); // warm-up
      gpu.device.queue.writeBuffer(counter, 0, new Uint32Array(1)); // cudaMemset
      const t = now();
      submit([{ pipe, group, grid: [blocks, 1, 1] }]);
      await done();
      out[key].gpuMs = now() - t;
      out[key].gpu = (await download(counter, 4, Uint32Array))[0];
      free(counter);
    }
  }
  return { len, blocks, threads, ...out };
}
function piOf(d, key, src) {
  const raw = d[key][src];
  const inCircle = key === 'opt' ? d.len - raw : raw;
  return (inCircle / d.len) * 4;
}
function lab5Console(d, key, src, ms) {
  const pi = piOf(d, key, src), PI = Math.PI;
  return `${key === 'opt' ? 'Optimized' : 'Unoptimized'}:\n PI (c var.): ${PI.toFixed(26)}\n PI result: ${pi.toFixed(26)}\n Defference: ${Math.abs(PI - pi).toFixed(26)}\n Ratio: ${(PI / pi).toFixed(26)}\n Time: ${fixed(ms, 6)} ms.`;
}
function renderLab5(d) {
  const src = gpuOk ? 'gpu' : 'cpu';
  const card = (key, title, what) => {
    const e = d[key], pi = piOf(d, key, src), diff = gpuOk ? Math.abs(e.gpu - e.cpu) : 0;
    return `<div class="panel" style="margin:0;padding:14px"><h2 style="margin-bottom:8px">${title} <small>· ${what}</small></h2><div class="stats" style="margin:0">${stat(pi.toFixed(6), 'π estimate')}${stat(fmt(Math.abs(Math.PI - pi), 3), '|π − estimate|')}${stat(`${(src === 'gpu' ? e.gpu : e.cpu).toLocaleString('en')}`, 'atomicAdd calls')}${stat(gpuOk ? msFmt(e.gpuMs) : 'n/a', 'GPU kernel')}${stat(msFmt(e.cpuMs), 'CPU')}${gpuOk ? stat(diff ? `${diff} points` : 'same count', 'GPU vs CPU', diff <= d.len * 1e-4 ? 'good' : 'bad') : ''}</div></div>`;
  };
  const xs = [], ys = [];
  let inside = 0;
  const step = Math.max(1, Math.floor(d.len / 300));
  for (let i = 0; i < d.len; i++) {
    const [x, y] = cpu.lab5Point(i);
    if (x * x + y * y <= 1) inside++;
    if ((i + 1) % step === 0 || i === d.len - 1) { xs.push(i + 1); ys.push((inside / (i + 1)) * 4); }
  }
  return `${engineNote()}
    <div class="algo-cards">${card('opt', 'Optimized', 'counts the points outside the circle')}${card('unopt', 'Unoptimized', 'counts the points inside')}</div>
    <div class="stats">${launch(`<<<${d.blocks}, ${d.threads}>>>, ${d.len} points`)}</div>
    <div class="two-col">
      <div data-copyable data-title="Monte Carlo points"><div class="copy-head"><h2>Sample points <small>· ${Math.min(d.len, 80000).toLocaleString('en')} of the unit square</small></h2>${actions('png')}</div><canvas class="scatter" width="720" height="720"></canvas>
        <div class="legend"><span><span class="sw" style="background:var(--c1)"></span>inside, x² + y² ≤ 1</span><span><span class="sw" style="background:var(--c2)"></span>outside: what the optimized kernel counts</span></div></div>
      <div>${chart({ title: 'π estimate as points accumulate (f64)', x: 'points', y: 'π estimate', series: [{ name: 'estimate', xs, ys, color: '--c1' }, { name: 'π', xs: [xs[0], xs[xs.length - 1]], ys: [Math.PI, Math.PI], color: '--muted', dash: true }] })}</div>
    </div>
    ${consoleBox(`${lab5Console(d, 'opt', src, src === 'gpu' ? d.opt.gpuMs : d.opt.cpuMs)}\n\n${lab5Console(d, 'unopt', src, src === 'gpu' ? d.unopt.gpuMs : d.unopt.cpuMs)}`)}`;
}
function drawScatter(canvas, len) {
  const S = canvas.width, ctx = canvas.getContext('2d');
  ctx.fillStyle = cssVar('--bg'); ctx.fillRect(0, 0, S, S);
  const cin = cssVar('--c1'), cout = cssVar('--c2');
  const n = Math.min(len, 80000);
  for (const inside of [true, false]) {
    ctx.fillStyle = inside ? cin : cout;
    for (let i = 0; i < n; i++) {
      const [x, y] = cpu.lab5Point(i);
      if ((x * x + y * y <= 1) !== inside) continue;
      ctx.fillRect(x * S - 1, (1 - y) * S - 1, 2, 2);
    }
  }
  ctx.strokeStyle = cssVar('--ink'); ctx.lineWidth = 3;
  ctx.beginPath(); ctx.arc(0, S, S, -Math.PI / 2, 0); ctx.stroke();
}

// ------------------------------------------------------------ control task --
async function runControl(p) {
  const layers = [p.len, ...parseLayers(p.layers)];
  const input = cpu.controlInput(p.len, p.seed);
  await pause();
  let t = now();
  const ref = cpu.controlCpu(input, layers);
  const cpuMs = now() - t;
  let steps = null, gpuMs = null;
  if (await gpuReady) {
    const pipe = await pipeline('control.wgsl', 'layerKernel');
    const layer = async (sum, outLen) => {
      const sumD = storage(new Float64Array([sum]));
      const outD = storage(outLen * 4);
      submit([{ pipe, group: bindGroup(pipe, { 0: outD, 1: sumD }), grid: [outLen, 1, 1] }]);
      const out = await download(outD, outLen * 4);
      free(sumD, outD);
      return out;
    };
    await layer(1, 1); // warm-up
    t = now();
    steps = [];
    let arr = input;
    for (let i = 1; i < layers.length; i++) {
      let sum = 0; // calcSum on the host, in double
      for (let k = 0; k < layers[i - 1]; k++) sum += arr[k];
      arr = await layer(sum, layers[i]);
      steps.push({ sum, seed: Math.trunc(sum), out: arr });
    }
    gpuMs = now() - t;
  }
  return { layers, input, ref, steps, gpuMs, cpuMs };
}
const parseLayers = (s) => String(s).split(/[\s,;]+/).filter(Boolean).map(Number);
function controlConsole(d, steps) {
  const show = (a) => Array.from(a, (v) => fixed(v, 6)).join('\n');
  const last = steps[steps.length - 1].out;
  return `Input array:\n${show(d.input)}\nOutput array:\n${show(last)}\n\nDuration: ${fixed(gpuOk ? d.gpuMs : d.cpuMs, 6)} ms.`;
}
function renderControl(d) {
  const steps = d.steps || d.ref.steps;
  const diverged = d.steps ? d.steps.findIndex((s, i) => s.seed !== d.ref.steps[i].seed) : -1;
  const outGpu = steps[steps.length - 1].out, outCpu = d.ref.output;
  const maxDiff = d.steps ? Math.max(...Array.from(outCpu, (v, i) => Math.abs(v - outGpu[i]))) : 0;
  const rows = steps.map((s, i) => {
    const r = d.ref.steps[i], off = d.steps && s.seed !== r.seed, n = d.layers[i + 1];
    const seeds = n > 1 ? `${s.seed} … ${s.seed + n - 1}` : `${s.seed}`;
    return `<tr${off ? ' class="off"' : ''}><td class="num">${i + 1}</td><td class="num">${d.layers[i]} → ${n}</td><td class="num">${s.sum.toFixed(7)}</td><td class="num">${seeds}</td><td class="num">${fixed(mean(s.out), 7)}</td><td class="num">${r.sum.toFixed(7)}</td><td class="num">${fixed(mean(r.out), 7)}</td><td class="num">${d.steps ? fmt(Math.max(...Array.from(r.out, (v, k) => Math.abs(v - s.out[k]))), 2) : "—"}</td></tr>`;
  }).join('');
  const xs = range(steps.length, (i) => i + 1);
  const note = diverged >= 0
    ? `<p class="note">The GPU path takes a different seed from layer ${diverged + 1}. Its outputs are f32, and <code>tanh</code> of a large argument rounds to exactly 1.0 in f32 but stays just below 1 in double. The host sum then lands on a whole number instead of just under it, so <code>(unsigned long long)sum</code>, the curand seed, is one higher, and every later layer draws other random weights.</p>`
    : '';
  return `${engineNote()}${note}
    <div class="stats">${stat(outGpu.every((v) => v === outGpu[0]) ? `${outGpu.length} × ${fixed(outGpu[0], 6)}` : Array.from(outGpu, (v) => fixed(v, 4)).join(', '), 'output array')}${timeStats(d.gpuMs, d.cpuMs)}${d.steps ? verdict(diverged < 0 && maxDiff < 1e-5, `max |Δ| ${fmt(maxDiff, 2)}`, diverged >= 0 ? `seed differs at layer ${diverged + 1}` : `max |Δ| ${fmt(maxDiff, 2)}`) : ''}${launch(`${steps.length} layers × <<<n, 1>>>`)}</div>
    <p class="hint" style="margin:-6px 0 14px">Thread <code>i</code> seeds curand with <code>(unsigned long long)sum + i</code>, so each output of a layer gets its own weight.</p>
    <div data-copyable data-title="Layer pipeline"><div class="copy-head"><h2>Pipeline <small>· one column per layer, cells shaded by value</small></h2>${actions('png')}</div><div class="plot layers-plot">${layersSvg(d, steps)}</div></div>
    <div class="charts two" style="margin-top:16px">
      ${chart({ title: 'Host sum fed into each layer', x: 'layer', y: 'sum', series: [{ name: 'GPU path', xs, ys: steps.map((s) => s.sum), color: '--c1' }, ...(d.steps ? [{ name: 'CPU f64', xs, ys: d.ref.steps.map((s) => s.sum), color: '--c2', dash: true }] : [])] })}
      ${chart({ title: 'Mean layer output tanh(weight · sum + 1)', x: 'layer', y: 'mean output', series: [{ name: 'GPU path', xs, ys: steps.map((s) => mean(s.out)), color: '--c1' }, ...(d.steps ? [{ name: 'CPU f64', xs, ys: d.ref.steps.map((s) => mean(s.out)), color: '--c2', dash: true }] : [])] })}
    </div>
    <div data-copyable style="margin-top:16px"><div class="copy-head"><h2>Layers <small>· ${d.steps ? 'GPU path, then the CPU f64 reference' : 'CPU f64 reference'}</small></h2>${actions('table')}</div>
      <div class="table-wrap"><table><thead><tr><th class="num">Layer</th><th class="num">Size</th><th class="num">sum</th><th class="num">seeds</th><th class="num">mean output</th><th class="num">CPU sum</th><th class="num">CPU mean output</th><th class="num">max |Δ|</th></tr></thead><tbody>${rows}</tbody></table></div></div>
    ${consoleBox(controlConsole(d, steps))}`;
}
function layersSvg(d, steps) {
  const cols = [d.input, ...steps.map((s) => s.out)];
  const W = 760, H = 250, top = 10, bottom = 34, colW = 34;
  const gap = (W - 40 - cols.length * colW) / Math.max(1, cols.length - 1);
  const x = (i) => 20 + i * (colW + gap);
  const parts = cols.map((a, i) => {
    const n = a.length, ch = (H - top - bottom) / n, y0 = top;
    const cells = Array.from(a, (v, k) => `<rect x="${x(i)}" y="${(y0 + k * ch).toFixed(2)}" width="${colW}" height="${Math.max(0.6, ch - (ch > 3 ? 1 : 0)).toFixed(2)}" fill="var(--c1)" fill-opacity="${(0.12 + 0.88 * Math.max(0, Math.min(1, v))).toFixed(3)}"><title>layer ${i}, #${k}: ${fixed(v, 6)}</title></rect>`).join('');
    const link = i ? `<path d="M${x(i - 1) + colW + 3} ${H / 2 - 8} L${x(i) - 3} ${H / 2 - 8}" stroke="var(--muted)" stroke-width="1.4" marker-end="url(#arr)"/>` : '';
    return `${link}${cells}<text x="${x(i) + colW / 2}" y="${H - bottom + 16}" text-anchor="middle">${n}</text><text x="${x(i) + colW / 2}" y="${H - 4}" text-anchor="middle">${i ? `L${i}` : 'in'}</text>`;
  }).join('');
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Layer pipeline"><defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L8 4L0 8z" fill="var(--muted)"/></marker></defs>${parts}</svg>`;
}

// -------------------------------------------------------------- lab config --
const LABS = [
  {
    id: 'lab1', tab: '1', name: 'Matrix multiplication', dir: '1-lab',
    intro: 'Multiplies two 211×211 integer matrices with the <code>matrixMult</code> kernel: a 2D grid of 211×211 blocks with one thread each, every thread computing one element of C. The result is checked element by element against a CPU triple loop, as the lab does.',
    port: 'The lab reads A and B from a file it first fills with <code>rand() % 1000</code>; here they come straight from the same generator (MSVC <code>rand()</code>, seed 1, since the lab never calls <code>srand</code>). Like the lab\'s timer, which stops after <code>cudaDeviceSynchronize()</code>, the GPU time waits for the kernel to finish.',
    fields: [
      { key: 'size', label: 'MATRIX_SIZE', value: 211, min: 2, max: 512 },
      { key: 'seed', label: 'srand seed', value: 1, min: 0, max: 4294967295 },
    ],
    run: runLab1, render: renderLab1,
    after: (out) => {
      out.querySelectorAll('[data-matrix]').forEach((box) => {
        previewMatrix(box);
        box.querySelectorAll('.tab').forEach((t) => (t.onclick = () => { box.dataset.which = t.dataset.pick; previewMatrix(box); }));
      });
      out.querySelectorAll('canvas.heat').forEach(drawHeat);
    },
    theme: (out) => out.querySelectorAll('canvas.heat').forEach(drawHeat),
  },
  {
    id: 'lab2', tab: '2', name: 'Animated function plot', dir: '2-lab',
    intro: 'Each frame the kernel decides, for every pixel of a 480×360 bitmap, whether the curve <i>y</i> = tan(sin <i>x</i> + cos <i>x</i>) passes within one pixel, and the host copies the bitmap into a <code>CPUAnimBitmap</code> window. The x range grows by DT every tick, so the curve slowly compresses.',
    port: 'GLUT and <code>CPUAnimBitmap</code> become a canvas: the compute shader fills a storage buffer, which is read back and drawn bottom-up, like <code>glDrawPixels</code>. The lab sizes its blocks from <code>maxThreadsPerBlock</code>; the port uses the device\'s <code>maxComputeInvocationsPerWorkgroup</code> the same way. <code>H / (MAXY − MINY)</code> is integer division in the lab (22, not 22.5) and stays so.',
    fields: [
      { key: 'W', label: 'W', value: 480, min: 16, max: 1920 },
      { key: 'H', label: 'H', value: 360, min: 16, max: 1080 },
      { key: 'MAXX', label: 'MAXX', value: 10, min: 1, max: 1000 },
      { key: 'MAXY', label: 'MAXY', value: 8, min: -100, max: 100 },
      { key: 'MINY', label: 'MINY', value: -8, min: -100, max: 100 },
      { key: 'DT', label: 'DT', value: 0.1, min: 0, max: 10, step: 'any' },
      { key: 'frames', label: 'Benchmark frames', value: 30, min: 1, max: 600 },
    ],
    hint: 'Run renders the benchmark frames on the GPU and on the CPU, compares the last frame pixel by pixel, then keeps animating.',
    run: runLab2, render: renderLab2, after: bindLab2,
  },
  {
    id: 'lab3', tab: '3', name: 'Shared-memory reduction', dir: '3-lab',
    intro: 'Six sums over 10 000 terms at once: A = Σ 1/i, B = the same terms in reverse, C and D split into even and odd terms. Each block of 1024 threads stores its terms in a <code>__shared__</code> cache, halves it with a tree reduction between <code>__syncthreads()</code>, and writes one partial sum per block; the host adds the 10 partial sums.',
    port: '<code>__shared__</code> becomes <code>var&lt;workgroup&gt;</code> and <code>__syncthreads()</code> becomes <code>workgroupBarrier()</code>. WGSL has no double, so the GPU sums are f32 and the CPU reference keeps f64.',
    fields: [
      { key: 'len', label: 'ARRAY_LENGTH', value: 10000, min: 1, max: 1000000 },
      { key: 'blocks', label: 'BLOCKS', value: 10, min: 1, max: 4096 },
      { key: 'tpb', label: 'THREADS_PER_BLOCK', value: '1024', options: ['64', '128', '256', '512', '1024'] },
    ],
    run: runLab3, render: renderLab3,
  },
  {
    id: 'lab4', tab: '4', name: 'Constant vs global memory', dir: '4-lab',
    intro: 'The same kernel runs twice on 512 random floats: once reading its 17 coefficients from global memory, once from <code>__constant__</code> memory filled with <code>cudaMemcpyToSymbol</code>. The lab times each run (allocate, copy, launch, copy back) with CUDA events.',
    port: '<code>__constant__</code> becomes a uniform buffer and the global array a read-only storage buffer. A single 512-thread launch takes microseconds, so each run here launches the kernel many times to make the difference measurable.',
    fields: [
      { key: 'size', label: 'ARR_SIZE', value: 512, min: 16, max: 1048576 },
      { key: 'repeats', label: 'Launches per run', value: 200, min: 1, max: 5000 },
      { key: 'runs', label: 'Runs', value: 10, min: 1, max: 100 },
      { key: 'seed', label: 'srand seed', value: 1, min: 0, max: 4294967295 },
    ],
    run: runLab4, render: renderLab4,
  },
  {
    id: 'lab5', tab: '5', name: 'Monte Carlo π', dir: '5-lab',
    intro: 'Estimates π from 80 000 points in the unit square. A grid-stride loop over 60 × 256 threads seeds a curand state with each point\'s index, draws x and y, and counts with <code>atomicAdd</code>. The optimized kernel counts the points outside the circle, about a fifth as many atomic operations; π = 4 · inside / total either way.',
    port: 'curand becomes a WGSL port of cuRAND\'s default XORWOW generator with the same <code>curand_init</code> scrambling, so each point index gets the same random numbers as on CUDA (rounded to f32, since WGSL has no double). <code>atomicAdd</code> becomes <code>atomicAdd</code> on an <code>atomic&lt;u32&gt;</code>. The lab\'s <code>main</code> runs only the optimized kernel; both run here.',
    fields: [
      { key: 'len', label: 'ARR_LENGTH', value: 80000, min: 1, max: 20000000 },
      { key: 'blocks', label: 'BLOCKS', value: 60, min: 1, max: 65535 },
      { key: 'threads', label: 'THREADS', value: 256, min: 1, max: 1024 },
    ],
    run: runLab5, render: renderLab5,
    after: (out, d) => out.querySelectorAll('canvas.scatter').forEach((c) => drawScatter(c, d.len)),
    theme: (out, d) => out.querySelectorAll('canvas.scatter').forEach((c) => drawScatter(c, d.len)),
  },
  {
    id: 'control', tab: 'CT', name: 'Multi-layer tanh pipeline', dir: 'control-task', title: 'Control task',
    intro: 'A random array of 10 values goes through 10 layers of sizes 10, 20, 40, 60, 120, 120, 60, 40, 20 and 4. For each layer the host sums the previous output and launches one single-thread block per output; thread <i>i</i> seeds curand with the sum plus <i>i</i>, draws a weight and writes tanh(weight · sum + 1).',
    port: 'The kernel gets the host\'s double sum as its raw 64 bits: it truncates them exactly for the seed and rounds them to f32 for the arithmetic. <code>tanh</code> is the lab\'s own exp formula, in f32 with a clamp so <code>exp</code> can\'t overflow.',
    fields: [
      { key: 'len', label: 'ARR_LENGTH', value: 10, min: 1, max: 10000 },
      { key: 'layers', label: 'Layer sizes', value: '10, 20, 40, 60, 120, 120, 60, 40, 20, 4', type: 'text', wide: true },
      { key: 'seed', label: 'srand seed', value: 1, min: 0, max: 4294967295 },
    ],
    run: runControl, render: renderControl,
  },
];

const consoleBox = (text) => `<div data-copyable style="margin-top:16px"><div class="copy-head"><h2>Console output <small>· in the lab's format</small></h2>${actions('text')}</div><pre class="console">${esc(text)}</pre></div>`;

// ------------------------------------------------------------------ copying --
const ACTION_LABEL = { png: 'Copy image', csv: 'Copy CSV', table: 'Copy table', results: 'Copy results', text: 'Copy text', code: 'Copy code', tsv: 'Copy TSV', download: 'Download .tsv' };
const ACTION_TIP = {
  png: 'Copy as a PNG image (downloads it if the clipboard is unavailable)', csv: 'Copy the plotted data as CSV', table: 'Copy the table as tab-separated text, pastes into spreadsheets',
  results: 'Copy the inputs and results as text', text: 'Copy the console output', code: 'Copy the source', tsv: 'Copy the whole matrix as tab-separated text', download: 'Download the whole matrix as a .tsv file',
};
const actions = (...kinds) => `<span class="copy-actions">${kinds.map((k) => `<button type="button" class="mini" data-act="${k}" title="${ACTION_TIP[k]}">${ACTION_LABEL[k]}</button>`).join('')}</span>`;

// Render an on-screen SVG to PNG: computed colours and fonts are inlined so the image
// matches the current theme, with the caption drawn above it.
function svgPng(svg, title, legend = [], scale = 2) {
  const box = svg.viewBox.baseVal;
  const w = (box && box.width) || +svg.getAttribute('width') || svg.clientWidth;
  const h = (box && box.height) || +svg.getAttribute('height') || svg.clientHeight;
  const clone = svg.cloneNode(true);
  const src = [svg, ...svg.querySelectorAll('*')], dst = [clone, ...clone.querySelectorAll('*')];
  src.forEach((el, i) => {
    const cs = getComputedStyle(el), c = dst[i];
    for (const a of ['fill', 'stroke', 'stroke-opacity', 'fill-opacity', 'stroke-width', 'opacity', 'visibility']) c.setAttribute(a, cs.getPropertyValue(a));
    if (el.tagName === 'text') for (const a of ['font-family', 'font-size', 'font-weight']) c.setAttribute(a, cs.getPropertyValue(a));
    c.removeAttribute('style'); c.removeAttribute('class');
  });
  clone.querySelectorAll('title, .marks').forEach((t) => t.remove());
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  clone.setAttribute('width', w); clone.setAttribute('height', h);
  if (!clone.getAttribute('viewBox')) clone.setAttribute('viewBox', `0 0 ${w} ${h}`);
  return framePng((ctx, x, y) => new Promise((resolve, reject) => {
    const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(clone)], { type: 'image/svg+xml' }));
    const img = new Image();
    img.onload = () => { ctx.drawImage(img, x, y, w, h); URL.revokeObjectURL(url); resolve(); };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('SVG render failed')); };
    img.src = url;
  }), w, h, title, legend, scale);
}
// A canvas drawing, framed the same way (canvas.toBlob does the encoding)
const canvasPng = (canvas, title) => {
  const w = Math.min(canvas.width, 720), h = (canvas.height / canvas.width) * w;
  return framePng((ctx, x, y) => { ctx.imageSmoothingEnabled = canvas.classList.contains('heat') ? false : true; ctx.drawImage(canvas, x, y, w, h); }, w, h, title, [], canvas.width >= 600 ? 1 : 2);
};
async function framePng(paint, w, h, title, legend, scale) {
  const css = getComputedStyle(document.documentElement);
  const head = title ? 30 : 0, pad = 14;
  const c = document.createElement('canvas');
  c.width = (w + pad * 2) * scale; c.height = (h + head + pad * 2) * scale;
  const ctx = c.getContext('2d');
  ctx.scale(scale, scale);
  ctx.fillStyle = css.getPropertyValue('--panel').trim();
  ctx.fillRect(0, 0, w + pad * 2, h + head + pad * 2);
  const family = getComputedStyle(document.body).fontFamily;
  if (title) {
    ctx.fillStyle = css.getPropertyValue('--ink').trim();
    ctx.font = `600 14px ${family}`;
    ctx.fillText(title, pad, pad + 14);
    ctx.font = `12px ${family}`;
    let x = w + pad;
    for (const { name, color } of [...legend].reverse()) {
      x -= ctx.measureText(name).width;
      ctx.fillStyle = css.getPropertyValue('--muted').trim();
      ctx.fillText(name, x, pad + 13);
      x -= 19;
      ctx.fillStyle = css.getPropertyValue(color).trim();
      ctx.fillRect(x, pad + 7, 14, 3);
      x -= 14;
    }
  }
  await paint(ctx, pad, pad + head);
  return new Promise((resolve, reject) => c.toBlob((b) => (b ? resolve(b) : reject(new Error('PNG encoding failed'))), 'image/png'));
}

const csvCell = (v) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
function chartCsv(spec) {
  const s0 = spec.series[0];
  const shared = spec.series.every((s) => s.xs.length === s0.xs.length && s.xs.every((x, i) => x === s0.xs[i]));
  if (shared) return [[spec.x, ...spec.series.map((s) => s.name || spec.y)], ...s0.xs.map((x, i) => [x, ...spec.series.map((s) => s.ys[i])])].map((r) => r.map(csvCell).join(',')).join('\n');
  return [['series', spec.x, spec.y], ...spec.series.flatMap((s) => s.xs.map((x, i) => [s.name, x, s.ys[i]]))].map((r) => r.map(csvCell).join(',')).join('\n');
}
const tableTsv = (table) => [...table.rows].map((r) => [...r.cells].map((c) => c.innerText.trim().replace(/\s+/g, ' ')).join('\t')).join('\n');
const labTitle = (lab) => lab.title || `Lab ${lab.tab}`;

function resultsText(sec) {
  const lab = LABS.find((l) => `panel-${l.id}` === sec.id);
  const p = params(lab);
  const lines = [`${labTitle(lab)}: ${lab.name}`, `Parameters: ${lab.fields.map((f) => `${f.label} = ${p[f.key]}`).join(', ')}`,
    `Engine: ${gpuOk ? `WebGPU (${gpu.info}) and the JavaScript CPU reference` : 'JavaScript CPU reference (no WebGPU)'}`];
  const out = $('.out', sec);
  out.querySelectorAll('.big, .note').forEach((el) => lines.push(el.innerText.trim()));
  out.querySelectorAll('.stats').forEach((st) => {
    const group = st.closest('.panel:not([role])')?.querySelector('h2')?.innerText.trim();
    if (group) lines.push(`${group}:`);
    st.querySelectorAll('.stat').forEach((x) => lines.push(`${group ? '  ' : ''}${$('span', x).innerText.trim()}: ${$('b', x).innerText.trim()}`));
  });
  out.querySelectorAll('pre.console').forEach((el) => lines.push('', 'Console output:', el.textContent));
  return lines.join('\n');
}

function downloadBlob(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

async function copyAction(btn) {
  const kind = btn.dataset.act, box = btn.closest('[data-copyable]');
  const done_ = (text) => { btn.textContent = text; setTimeout(() => (btn.textContent = ACTION_LABEL[kind]), 1400); };
  const name = (box?.dataset.title || 'chart').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  try {
    if (kind === 'png') {
      const canvas = box.querySelector('canvas');
      const svg = box.querySelector('.plot svg');
      const spec = charts.get(box.dataset.chart);
      const legend = spec && spec.series.length > 1 ? spec.series.map((x) => ({ name: x.name, color: x.color })) : [];
      const make = () => (canvas && !svg ? canvasPng(canvas, box.dataset.title) : svgPng(svg, box.dataset.title, legend));
      try {
        // pass the promise straight to ClipboardItem so Safari keeps the user gesture
        await navigator.clipboard.write([new ClipboardItem({ 'image/png': make() })]);
        return done_('Copied');
      } catch {
        downloadBlob(await make(), `${name}.png`);
        return done_('Downloaded');
      }
    }
    if (kind === 'download' || kind === 'tsv') {
      const m = matrices.get(box.dataset.matrix), which = box.dataset.which;
      const text = matrixTsv(m, which);
      if (kind === 'download') { downloadBlob(new Blob([text + '\n'], { type: 'text/tab-separated-values' }), `matrix-${which}-${m.N}x${m.N}.tsv`); return done_('Downloaded'); }
      await navigator.clipboard.writeText(text);
      return done_('Copied');
    }
    const text = kind === 'csv' ? chartCsv(charts.get(box.dataset.chart))
      : kind === 'table' ? tableTsv(box.querySelector('table'))
      : kind === 'text' || kind === 'code' ? box.querySelector('pre').textContent
      : resultsText(btn.closest('section.panel'));
    await navigator.clipboard.writeText(text);
    done_('Copied');
  } catch (e) { console.error(e); done_('Failed'); }
}
document.addEventListener('click', (e) => { const b = e.target.closest('button[data-act]'); if (b) copyAction(b); });

// -------------------------------------------------------------- code view --
const CU_KW = /\b(__global__|__device__|__host__|__shared__|__constant__|__syncthreads|atomicAdd|threadIdx|blockIdx|blockDim|gridDim|cudaMalloc|cudaMemcpy|cudaMemcpyToSymbol|cudaFree|cudaMemset|cudaEvent\w*|curand\w*|dim3)\b/;
const WG_KW = /\b(var<workgroup>|workgroupBarrier|atomicAdd|atomic|@compute|@workgroup_size|@builtin|@group|@binding|override|uniform|storage)\b/;
const C_KW = /\b(if|else|for|while|return|const|let|var|fn|struct|int|unsigned|char|bool|float|double|void|typedef|true|false|auto|new|delete|sizeof|continue|break|u32|i32|f32|vec2|vec3|vec4|array|ptr|select|using|namespace|include|define)\b/;
function highlight(code, lang) {
  const re = /(\/\/[^\n]*|\/\*[\s\S]*?\*\/)|("(?:[^"\\\n]|\\.)*")|(^[ \t]*#\w+)|(@\w+|var<\w+>|\b[A-Za-z_]\w*\b)|(\b\d[\d.]*(?:e-?\d+)?[uUfiL]*\b|\b0x[\da-fA-F]+u?\b)/gm;
  const special = lang === 'cu' ? CU_KW : WG_KW;
  let out = '', last = 0;
  for (const m of code.matchAll(re)) {
    out += esc(code.slice(last, m.index));
    const [t] = m;
    const cls = m[1] ? 'cm' : m[2] ? 'st' : m[3] ? 'pp' : m[5] ? 'nu' : special.test(t) && t.match(special)[0] === t ? 'gp' : C_KW.test(t) && t.match(C_KW)[0] === t ? 'kw' : '';
    out += cls ? `<span class="${cls}">${esc(t)}</span>` : esc(t);
    last = m.index + t.length;
  }
  return out + esc(code.slice(last));
}
const CODE = {
  lab1: ['1-lab/kernel.cu', 'lab1.wgsl'], lab2: ['2-lab/kernel.cu', 'lab2.wgsl'], lab3: ['3-lab/kernel.cu', 'lab3.wgsl'],
  lab4: ['4-lab/kernel.cu', 'lab4.wgsl'], lab5: ['5-lab/kernel.cu', 'lab5.wgsl'], control: ['control-task/kernel.cu', 'control.wgsl'],
};
async function loadCode(det, lab) {
  if (det.dataset.loaded) return;
  det.dataset.loaded = '1';
  const [cu, wgsl] = CODE[lab.id];
  const fill = async (sel, getText, lang) => {
    const pre = $(sel, det);
    try { pre.innerHTML = highlight((await getText()).replace(/^﻿/, '').replace(/\t/g, '    '), lang); }
    catch (e) { pre.textContent = `Couldn't load the source: ${e.message}`; }
  };
  await Promise.all([
    fill('.cu pre', async () => { const r = await fetch(`../${cu}`); if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.text(); }, 'cu'),
    fill('.wg pre', () => wgslSource(wgsl), 'wgsl'),
  ]);
}
const codeView = (lab) => {
  const [cu, wgsl] = CODE[lab.id];
  return `<details class="code"><summary>Kernel source: the CUDA original next to its WGSL port</summary>
    <div class="code-grid">
      <div class="cu" data-copyable><div class="copy-head"><h2><a href="${SRC}/${cu}" target="_blank" rel="noopener">${cu}</a></h2>${actions('code')}</div><pre class="src">Loading…</pre></div>
      <div class="wg" data-copyable><div class="copy-head"><h2><a href="${SRC}/web/kernels/${wgsl}" target="_blank" rel="noopener">web/kernels/${wgsl}</a></h2>${actions('code')}</div><pre class="src">Loading…</pre></div>
    </div></details>`;
};

// --------------------------------------------------------------- framework --
const nav = $('#nav'), main = $('#labs');
const state = new Map(); // lab id -> { ran, running, data }

function fieldHtml(lab, f) {
  const id = `${lab.id}-${f.key}`;
  if (f.options) return `<label class="field" for="${id}">${esc(f.label)}<select id="${id}" name="${f.key}">${f.options.map((o) => { const [v, t] = Array.isArray(o) ? o : [o, o]; return `<option value="${v}"${v === f.value ? ' selected' : ''}>${t}</option>`; }).join('')}</select></label>`;
  const type = f.type || 'number';
  const attrs = type === 'number' ? ` min="${f.min}" max="${f.max}" step="${f.step || 1}" required` : ' spellcheck="false"';
  return `<label class="field${f.wide ? ' wide' : ''}" for="${id}">${esc(f.label)}<input id="${id}" name="${f.key}" type="${type}" value="${esc(f.value)}"${attrs}></label>`;
}

function build() {
  for (const lab of LABS) {
    const b = document.createElement('button');
    b.type = 'button'; b.role = 'tab'; b.dataset.lab = lab.id;
    b.setAttribute('aria-controls', `panel-${lab.id}`);
    b.innerHTML = `<span class="lang cu">CU</span>${lab.title || `Lab ${lab.tab}`}<span class="long muted" style="font-weight:400">· ${esc(lab.name)}</span>`;
    b.onclick = () => select(lab.id, true);
    nav.appendChild(b);

    const sec = document.createElement('section');
    sec.className = 'panel'; sec.id = `panel-${lab.id}`; sec.role = 'tabpanel'; sec.hidden = true;
    sec.innerHTML = `<div class="lab-head"><span class="lang cu">CUDA → WGSL</span><h3>${labTitle(lab)}: ${esc(lab.name)}</h3><a href="${SRC}/${lab.dir}/kernel.cu" target="_blank" rel="noopener">View source</a></div>
      <p class="intro">${lab.intro}</p>
      <p class="port"><b>Ported to WebGPU</b> because CUDA can't run in a browser. ${lab.port}</p>
      <form class="params" novalidate>${lab.fields.map((f) => fieldHtml(lab, f)).join('')}<button class="run" type="submit">Run</button></form>
      ${lab.hint ? `<p class="hint">${lab.hint}</p>` : ''}
      <div class="out" style="margin-top:18px"></div>
      ${codeView(lab)}`;
    main.appendChild(sec);
    state.set(lab.id, { ran: false, running: false, data: null });

    $('form', sec).addEventListener('submit', (e) => { e.preventDefault(); run(lab); });
    const det = $('details.code', sec);
    det.addEventListener('toggle', () => det.open && loadCode(det, lab));
  }
}

function params(lab) {
  const fd = new FormData($(`#panel-${lab.id} form`)), p = {};
  for (const f of lab.fields) { const v = fd.get(f.key); p[f.key] = f.type === 'text' || f.options ? v : Number(v); }
  return p;
}

function invalid(lab, p) {
  for (const f of lab.fields) {
    if (f.type === 'text' || f.options) continue;
    const v = p[f.key];
    if (!Number.isFinite(v) || v < f.min || v > f.max || (f.step !== 'any' && !Number.isInteger(v))) return `${f.label} must be ${f.step === 'any' ? 'a number' : 'an integer'} from ${f.min} to ${f.max}.`;
  }
  if (lab.id === 'lab2' && p.MAXY <= p.MINY) return 'MAXY must be greater than MINY.';
  if (lab.id === 'control') {
    const ls = parseLayers(p.layers);
    if (!ls.length || ls.length > 40 || ls.some((n) => !Number.isInteger(n) || n < 1 || n > 65535)) return 'Layer sizes must be 1 to 40 whole numbers from 1 to 65535, separated by commas.';
  }
  return null;
}

async function run(lab) {
  const st = state.get(lab.id);
  if (st.running) return;
  const sec = $(`#panel-${lab.id}`), btn = $('.run', sec), out = $('.out', sec);
  const p = params(lab);
  const bad = invalid(lab, p);
  if (bad) { out.innerHTML = `<div class="err">${esc(bad)}</div>`; return; }
  st.running = true; st.ran = true;
  btn.disabled = true; btn.textContent = 'Running…';
  out.style.opacity = '.45';
  try {
    const data = await lab.run(p);
    out.innerHTML = `<div class="out-actions">${actions('results')}</div>` + lab.render(data, p);
    st.data = data;
    drawCharts(out);
    lab.after?.(out, data);
  } catch (e) {
    console.error(e);
    out.innerHTML = `<div class="err">${esc(e.message || e)}</div>`;
  } finally {
    st.running = false; btn.disabled = false; btn.textContent = 'Run'; out.style.opacity = '';
  }
}

window.addEventListener('themechange', () => {
  for (const lab of LABS) {
    const st = state.get(lab.id);
    if (lab.theme && st.data) lab.theme($(`#panel-${lab.id} .out`), st.data);
  }
});

function select(id, push) {
  const lab = LABS.find((l) => l.id === id) || LABS[0];
  nav.querySelectorAll('button').forEach((b) => b.setAttribute('aria-selected', b.dataset.lab === lab.id));
  main.querySelectorAll('section.panel[role="tabpanel"]').forEach((s) => (s.hidden = s.id !== `panel-${lab.id}`));
  if (push) try { history.replaceState(null, '', `#${lab.id}`); } catch {}
  document.title = `${labTitle(lab)} · ParallelCuda Labs`;
  if (lab.id !== 'lab2') stopAnim();
  if (!state.get(lab.id).ran) run(lab);
  else {
    drawCharts($(`#panel-${lab.id}`));
    if (lab.id === 'lab2') { const box = $('#panel-lab2 .anim-box'); if (box) startAnim(box); }
  }
}

function showGpuStatus() {
  const el = $('#gpu-status');
  if (gpuOk) {
    const l = gpu.limits;
    el.className = 'gpu-status ok-box';
    el.innerHTML = `<b>WebGPU</b> on ${esc(gpu.info)} · up to ${l.maxComputeInvocationsPerWorkgroup} threads per workgroup, ${l.maxComputeWorkgroupStorageSize / 1024} KB of workgroup memory. Every lab runs its kernel on the GPU and checks it against the CPU.`;
  } else {
    el.className = 'gpu-status warn-box';
    el.innerHTML = `<b>No WebGPU:</b> ${esc(gpu.reason)}. The labs run their JavaScript CPU reference only. A recent Chrome or Edge, or Safari 26 and later, runs the GPU kernels.`;
  }
  el.hidden = false;
}

build();
select(location.hash.slice(1), false);
window.addEventListener('hashchange', () => select(location.hash.slice(1), false));
