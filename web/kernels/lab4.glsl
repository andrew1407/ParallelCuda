// WebGL2 port of kernelGlobal and kernelConst from 4-lab/kernel.cu, one fragment per element.
// __constant__ float constArrConst[CONST_SIZE] becomes a uniform array (KERNEL_CONST); the
// global-memory copy is a 20 × 1 float texture read with texelFetch. A and B are R32F textures of
// W texels per row, element i at (i % W, i / W).

uniform sampler2D A;
uniform int ARR_SIZE;
uniform int W;

#ifdef KERNEL_CONST
uniform float constArrConst[17];
#define CONST(u) constArrConst[u]
#else
uniform sampler2D constArr;
#define CONST(u) texelFetch(constArr, ivec2(u, 0), 0).r
#endif

out float B;

void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  int i = p.x + p.y * W; // threadIdx.x + blockIdx.x * blockDim.x
  B = 0.0;
  if (i >= ARR_SIZE || i == 0) {
    return; // the formula divides by i, so B[0] stays 0
  }
  float a = texelFetch(A, p, 0).r;
  float fi = float(i);
  for (int u = 1; u <= 15; u += 2) {
    B += i != 15
      ? CONST(u) * a / fi - CONST(u + 1) * a * fi
      : CONST(0) * a * fi + CONST(15) * a / fi;
  }
}
