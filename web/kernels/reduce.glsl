// One step of a parallel sum in WebGL2, standing in for atomicAdd: each fragment adds a 2 × 2 tile
// of the previous level's u32 counts, so a W × H texture shrinks to 1 × 1 in log2 passes.

uniform usampler2D src;
uniform ivec2 size; // the previous level's size

out uint sum;

void main() {
  ivec2 p = ivec2(gl_FragCoord.xy) * 2;
  uint s = 0u;
  for (int dy = 0; dy < 2; dy++) {
    for (int dx = 0; dx < 2; dx++) {
      ivec2 q = p + ivec2(dx, dy);
      if (q.x < size.x && q.y < size.y) s += texelFetch(src, q, 0).r;
    }
  }
  sum = s;
}
