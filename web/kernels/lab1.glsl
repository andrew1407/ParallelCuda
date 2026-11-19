// WebGL2 port of matrixMult from 1-lab/kernel.cu, as fragment-shader GPGPU.
// <<<dim3(MATRIX_SIZE, MATRIX_SIZE), 1>>> becomes one MATRIX_SIZE × MATRIX_SIZE pass: the fragment
// at (x, y) is block (x, y), so it computes C[y][x]. A, B and C are R32I textures, row i at y = i.

uniform isampler2D matrixA;
uniform isampler2D matrixB;
uniform int MATRIX_SIZE;

out int matrixResult;

void main() {
  int j = int(gl_FragCoord.x); // blockDim.x * blockIdx.x + threadIdx.x
  int i = int(gl_FragCoord.y); // blockDim.y * blockIdx.y + threadIdx.y
  int resValue = 0;
  for (int k = 0; k < MATRIX_SIZE; k++) {
    resValue += texelFetch(matrixA, ivec2(k, i), 0).r * texelFetch(matrixB, ivec2(j, k), 0).r;
  }
  matrixResult = resValue;
}
