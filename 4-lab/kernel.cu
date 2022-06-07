#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#define ARR_SIZE 512
#define CONST_SIZE 16
#define SIZE_MEMORY(s) (s * sizeof(float))

float* constArrGlobal;
__constant__ float constArrConst[CONST_SIZE];


__global__ void kernelGlobal(float* A, float* B, float* constArr) {
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	B[i] = 0;
	for (int u = 1; u <= 15; u += 2)
		B[i] += (i != 15) ?
		(constArr[u] * A[i] / i - constArr[u + 1] * A[i] * i) :
		(constArr[0] * A[i] * i + constArr[15] * A[i] / i);
}

__global__ void kernelConst(float* A, float* B) {
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	B[i] = 0;
	for (int u = 1; u <= 15; u += 2)
		B[i] += (i != 15) ?
		(constArrConst[u] * A[i] / i - constArrConst[u + 1] * A[i] * i) :
		(constArrConst[0] * A[i] * i + constArrConst[15] * A[i] / i);
}


__host__ void fillArray(float* arr, const int size) {
	for (int i = 0; i < size; i++)
		arr[i] = ((float)rand()) / (float)RAND_MAX;
}

__host__ void execGlobal() {
	cudaEvent_t start, finish;
	cudaEventCreate(&start);
	cudaEventCreate(&finish);

	cudaEventRecord(start, 0);

	float* _constArr = new float[CONST_SIZE];
	float* Ah = new float[ARR_SIZE];
	float* Ad = nullptr;
	float* Bh = new float[ARR_SIZE];
	float* Bd = nullptr;

	fillArray(_constArr, CONST_SIZE);
	fillArray(Ah, ARR_SIZE);

	cudaMalloc(&Ad, SIZE_MEMORY(ARR_SIZE));
	cudaMalloc(&Bd, SIZE_MEMORY(ARR_SIZE));
	cudaMalloc(&constArrGlobal, SIZE_MEMORY(CONST_SIZE));

	cudaMemcpy(Ad, Ah, SIZE_MEMORY(ARR_SIZE), cudaMemcpyHostToDevice);
	cudaMemcpy(constArrGlobal, _constArr, SIZE_MEMORY(CONST_SIZE), cudaMemcpyHostToDevice);


	kernelGlobal <<<1, ARR_SIZE >>> (Ad, Bd, constArrGlobal);

	cudaMemcpy(Bh, Bd, SIZE_MEMORY(ARR_SIZE), cudaMemcpyDeviceToHost);

	cudaEventRecord(finish, 0);
	cudaEventSynchronize(finish);


	float duration;
	cudaEventElapsedTime(&duration, start, finish);
	std::cout << "Global: " << duration << " ms." << std::endl;

	delete[] _constArr;
	delete[] Ah;
	delete[] Bh;

	cudaFree(Ad);
	cudaFree(Bd);
	cudaFree(constArrGlobal);
}

__host__ void execConst() {
	cudaEvent_t start, finish;
	cudaEventCreate(&start);
	cudaEventCreate(&finish);

	cudaEventRecord(start, 0);

	float* _constArr = new float[CONST_SIZE];
	float* Ah = new float[ARR_SIZE];
	float* Ad = nullptr;
	float* Bh = new float[ARR_SIZE];
	float* Bd = nullptr;

	fillArray(_constArr, CONST_SIZE);
	fillArray(Ah, ARR_SIZE);

	cudaMalloc(&Ad, SIZE_MEMORY(ARR_SIZE));
	cudaMalloc(&Bd, SIZE_MEMORY(ARR_SIZE));

	cudaMemcpy(Ad, Ah, SIZE_MEMORY(ARR_SIZE), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(constArrConst, _constArr, SIZE_MEMORY(CONST_SIZE));


	kernelConst <<<1, ARR_SIZE>>> (Ad, Bd);

	cudaMemcpy(Bh, Bd, SIZE_MEMORY(ARR_SIZE), cudaMemcpyDeviceToHost);

	cudaEventRecord(finish, 0);
	cudaEventSynchronize(finish);

	float duration;
	cudaEventElapsedTime(&duration, start, finish);
	std::cout << "Constant: " << duration << " ms." << std::endl;

	delete[] _constArr;
	delete[] Ah;
	delete[] Bh;

	cudaFree(Ad);
	cudaFree(Bd);
}

int main() {
	execGlobal();
	execConst();

	return 0;
}
