// Runs the JavaScript CPU reference (cpu.js) off the main thread, so a big input never freezes the page.
// Each job times itself here, so the CPU timings don't include the message round trip.
import * as cpu from './cpu.js';

const now = () => performance.now();
const jobs = {
  lab1({ A, B, N }) {
    const t = now();
    const C = cpu.lab1Cpu(A, B, N);
    return { C, ms: now() - t };
  },
  lab2Bench({ p, frames }) {
    const px = new Uint32Array(p.W * p.H);
    const t = now();
    for (let k = 1; k <= frames; k++) cpu.lab2Frame(px, k, p);
    return { px, ms: (now() - t) / frames };
  },
  lab2Frame({ p, ticks }) {
    const px = new Uint32Array(p.W * p.H);
    cpu.lab2Frame(px, ticks, p);
    return { px };
  },
  lab3({ len, blocks, tpb }) {
    const t = now();
    const ref = cpu.lab3Cpu(len, blocks, tpb);
    return { ref, ms: now() - t };
  },
  lab4({ g, c, K }) {
    const t = now();
    for (let k = 0; k < K; k++) cpu.lab4Cpu(g.A, g.c);
    const ms = (now() - t) / K;
    return { ms, ref: { g: cpu.lab4Cpu(g.A, g.c), c: cpu.lab4Cpu(c.A, c.c) } };
  },
  lab5({ len }) {
    const out = {};
    for (const [key, reversed] of [['opt', true], ['unopt', false]]) {
      const t = now();
      const counter = cpu.lab5Cpu(len, reversed);
      out[key] = { cpu: counter, cpuMs: now() - t };
    }
    // the running π estimate for the chart, about 300 points
    const xs = [], ys = [];
    let inside = 0;
    const step = Math.max(1, Math.floor(len / 300));
    for (let i = 0; i < len; i++) {
      const [x, y] = cpu.lab5Point(i);
      if (x * x + y * y <= 1) inside++;
      if ((i + 1) % step === 0 || i === len - 1) { xs.push(i + 1); ys.push((inside / (i + 1)) * 4); }
    }
    return { ...out, series: { xs, ys } };
  },
  control({ input, layers }) {
    const t = now();
    const ref = cpu.controlCpu(input, layers);
    return { ref, ms: now() - t };
  },
};

const buffers = (v, acc = []) => {
  if (ArrayBuffer.isView(v)) { if (!acc.includes(v.buffer)) acc.push(v.buffer); }
  else if (v && typeof v === 'object') for (const x of Object.values(v)) buffers(x, acc);
  return acc;
};

self.onmessage = ({ data: { id, job, args } }) => {
  try {
    const r = jobs[job](args);
    self.postMessage({ id, r }, buffers(r));
  } catch (e) {
    self.postMessage({ id, error: String(e?.message || e) });
  }
};
