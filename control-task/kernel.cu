#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

#define ARR_LENGTH 10

__device__ double fn(double x) {
	const double expPlus = exp(x);
	const double expMinus = exp(-x);
	return (expPlus - expMinus) / (expPlus + expMinus);
}

__global__ void kernel(double* output, double* sum) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	curandState_t state;
	curand_init(*sum, 0, 0, &state);
	const double weight = curand_uniform_double(&state);
	const double a = weight * (*sum) + 1.0;
	output[i] = fn(a);
}

__host__ void fillArrRand(double* arr, const int length) {
	for (int i = 0; i < length; i++)
		arr[i] = (double)rand() / (double)RAND_MAX;
}

__host__ double calcSum(double* arr, const int length) {
	double res = 0.0;
	for (int i = 0; i < length; i++) res += arr[i];
	return res;
}

__host__ void showArr(double* arr, const int length) {
	for (int i = 0; i < length; i++)
		printf("%f\n", arr[i]);
}

int main() {
	const int layers[11] = { ARR_LENGTH, 10, 20, 40, 60, 120, 120, 60, 40, 20, 4 };
	double* arr_h = new double[ARR_LENGTH];
	fillArrRand(arr_h, ARR_LENGTH);
	double* arrIterable = arr_h;

	cudaEvent_t start, finish;
	cudaEventCreate(&start);
	cudaEventCreate(&finish);

	printf("Input array:\n");
	showArr(arr_h, ARR_LENGTH);

	cudaEventRecord(start, 0);

	for (int i = 1; i < 11; i++) {
		const int inputLength = layers[i - 1];
		const int outputLength = layers[i];
		double sum_h = calcSum(arrIterable, inputLength);

		const int inputSize = inputLength * sizeof(double);
		const int outputSize = outputLength * sizeof(double);

		double* sum_d = nullptr;
		double* output_d = nullptr;

		cudaMalloc(&sum_d, sizeof(double));
		cudaMalloc(&output_d, outputSize);

		cudaMemcpy(sum_d, &sum_h, sizeof(double), cudaMemcpyHostToDevice);

		kernel <<<outputLength, 1>>> (output_d, sum_d);

		double* output_h = new double[outputLength];
		cudaMemcpy(output_h, output_d, outputSize, cudaMemcpyDeviceToHost);

		cudaFree(output_d);
		cudaFree(&sum_d);

		delete[] arrIterable;
		arrIterable = output_h;
	}

	cudaEventRecord(finish, 0);
	cudaEventSynchronize(finish);

	printf("Output array:\n");
	showArr(arrIterable, layers[10]);

	float duration;
	cudaEventElapsedTime(&duration, start, finish);
	printf("\nDuration: %f ms.\n", duration);

	delete[] arrIterable;

	return 0;
}