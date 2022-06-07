#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cpu_anim.h"

// input parameters
#define W 480
#define H 360
#define MAXX 10
#define MAXY 8
#define MINY -8
#define DT 0.1

// bitmap data
struct DataBlock {
  unsigned char* dev_bitmap;
  CPUAnimBitmap* bitmap;
};

// calculating measures
struct GPU_parameters {
    dim3 threads, blocks;
} dimentions;

// calculating function
__device__ double fn(const double x) {
    return tan(sin(x) + cos(x));
};

// main computing function per frame
__global__ void kernel(unsigned char* ptr, int ticks)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;
  float c[3] = { 0.0, 0.5, -0.5 };
  bool lesserThanOne = false;
  for (float xDelta = -0.5; xDelta <= 0.5; xDelta += 0.5) {
      const double cX = (x + xDelta) / W * (MAXX + ticks * DT);
      const float c = abs(H / (MAXY - MINY) * (fn(cX) - MINY) - y);
      lesserThanOne = lesserThanOne || c <= 1;
  }
  const int offsetIndex = offset * 4;
  ptr[offsetIndex] = ptr[offsetIndex + 3] = 255;
  ptr[offsetIndex + 1] = ptr[offsetIndex + 2] = lesserThanOne ? 0 : 255;
}

// caclucating dimesnions setter
__host__ void setDimentions() {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int mtpb = deviceProp.maxThreadsPerBlock;
    const int threadsPerDim = ceil(sqrt(mtpb));
    const int Wdim = (W + threadsPerDim - 1) / threadsPerDim;
    const int Hdim = (H + threadsPerDim - 1) / threadsPerDim;
    dimentions.threads = dim3(threadsPerDim, threadsPerDim);
    dimentions.blocks = dim3(Wdim, Hdim);
}

// callback cleaner per frame
__host__ void cleanup(DataBlock* d)
{
  cudaFree(d->dev_bitmap);
}

// callback frame generator
__host__ void generate_frame(DataBlock* d, int ticks)
{
  kernel<<<dimentions.blocks, dimentions.threads>>>(d->dev_bitmap, ticks);
  cudaMemcpy(
      d->bitmap->get_ptr(),
      d->dev_bitmap,
      d->bitmap->image_size(),
      cudaMemcpyDeviceToHost
  );
}

int main() {
    setDimentions();    //define dimentions
    DataBlock data;
    CPUAnimBitmap bitmap(W, H, &data);
    data.bitmap = &bitmap;
    cudaMalloc(&data.dev_bitmap, bitmap.image_size());
    // infinite slideshow
    bitmap.anim_and_exit(
        (void (*) (void*, int)) generate_frame,
        (void(*) (void*)) cleanup
    );

    return 0;
}