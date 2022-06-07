#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_atomic_functions.h>

#include <curand.h>
#include <curand_kernel.h>

#include <stdio.h>

#define M_PI           3.14159265358979323846

#define BLOCKS 60
#define THREADS 256
#define ARR_LENGTH 80000


typedef struct { double x, y; } coords;

__device__ coords getCoordsByDist(int* i) {
	curandState_t state;
	curand_init(*i, 0, 0, &state);
	const double x = curand_uniform_double(&state);
	const double y = curand_uniform_double(&state);
	return { x, y };
}

__device__ void kernelExec(int* counter, const bool reversed) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int blockSize = blockDim.x * gridDim.x;
	while (i < ARR_LENGTH) {
		const coords dot = getCoordsByDist(&i);
		const double radius = pow(dot.x, 2) + pow(dot.y, 2);
		const bool matches = reversed ? (radius > 1.0) : (radius <= 1.0);
		if (matches) atomicAdd(counter, 1);
		i += blockSize;
	}
}

__global__ void kernelOptimized(int* counter) {
	kernelExec(counter, true);
}

__global__ void kernelUnoptimized(int* counter) {
	kernelExec(counter, false);
}

__host__ void showResult(const char* title, const int* included, const float* duration) {
	const double resultPI = (double)(*included) / (double)ARR_LENGTH * 4.0;
	char* resultStr = "%s:\n PI (c var.): %.26f\n PI result: %.26f\n Defference: %.26f\n Ratio: %.26f\n Time: %f ms.\n";
	printf(resultStr, title, M_PI, resultPI, abs(M_PI - resultPI), (double)M_PI / resultPI, *duration);
}

__host__ void calcPI(const bool optimized) {
	cudaEvent_t start, finish;
	cudaEventCreate(&start);
	cudaEventCreate(&finish);


	int* counter_d = nullptr;
	int* counter_h = new int(0);

	cudaMalloc(&counter_d, sizeof(int));
	cudaMemset(counter_d, 0, sizeof(int));

	cudaEventRecord(start, 0);

	if (optimized)
		kernelOptimized << <BLOCKS, THREADS >> > (counter_d);
	else
		kernelUnoptimized << <BLOCKS, THREADS >> > (counter_d);

	cudaEventRecord(finish, 0);
	cudaEventSynchronize(finish);

	cudaMemcpy(counter_h, counter_d, sizeof(int), cudaMemcpyDeviceToHost);

	float duration;
	cudaEventElapsedTime(&duration, start, finish);

	if (optimized) {
		const int inCircle = ARR_LENGTH - *counter_h;
		showResult("Optimized", &inCircle, &duration);
	} else {
		showResult("Unoptimized", counter_h, &duration);
	}

	delete counter_h;
	cudaFree(counter_d);

}

int main() {
	const bool optimized = true;
	calcPI(optimized);

	return 0;
}
