// Checks for WebGL2 with float render targets off the main thread. The first WebGL context
// can take a while to create (it may start the GPU process), and a worker doesn't block the page.
let reason = '';
try {
  const gl = typeof OffscreenCanvas === 'undefined' ? null : new OffscreenCanvas(1, 1).getContext('webgl2');
  if (!gl) reason = 'this browser has no WebGL2';
  else if (!gl.getExtension('EXT_color_buffer_float')) reason = 'this WebGL2 has no float render targets (EXT_color_buffer_float)';
  gl?.getExtension('WEBGL_lose_context')?.loseContext();
} catch (e) {
  reason = `WebGL2 failed to start (${e.message || e})`;
}
postMessage(reason);
