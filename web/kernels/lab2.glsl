// WebGL2 port of kernel from 2-lab/kernel.cu: renders one frame of tan(sin(x) + cos(x)) into an
// RGBA8 texture. A thread per pixel becomes a fragment per pixel; gl_FragCoord.y = 0 is the bottom
// row, like the bitmap the lab hands to glDrawPixels. GLSL has no double, so the math is f32.

uniform int W;
uniform int H;
uniform int MAXX;
uniform int MAXY;
uniform int MINY;
uniform float DT;
uniform int ticks;

out vec4 bitmap;

float func(float x) {
  return tan(sin(x) + cos(x));
}

void main() {
  int x = int(gl_FragCoord.x);
  int y = int(gl_FragCoord.y);
  bool lesserThanOne = false;
  for (float xDelta = -0.5; xDelta <= 0.5; xDelta += 0.5) {
    float cX = (float(x) + xDelta) / float(W) * (float(MAXX) + float(ticks) * DT);
    float c = abs(float(H / (MAXY - MINY)) * (func(cX) - float(MINY)) - float(y));
    lesserThanOne = lesserThanOne || c <= 1.0;
  }
  // ptr[offset * 4 + 0..3] = { 255, lesserThanOne ? 0 : 255, same, 255 }
  float gb = lesserThanOne ? 0.0 : 1.0;
  bitmap = vec4(1.0, gb, gb, 1.0);
}
