#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <functional>

typedef std::function<void(int)> lambdaIterator;

#define THREADS_PER_BLOCK 1024
#define BLOCKS 10
#define ARRAY_LENGTH 10000

//typedef float baseType;
typedef double baseType;

__global__ void calculateSums (
	baseType* A,
	baseType* B,
	baseType* C1,
	baseType* C2,
	baseType* D1,
	baseType* D2
) {
	const int sumsAmount = 6;
	__shared__ baseType sumsCache[sumsAmount][THREADS_PER_BLOCK];
	const int valIdx = threadIdx.x + blockIdx.x * blockDim.x;
	const int cacheIdx = threadIdx.x;
	if (valIdx >= ARRAY_LENGTH) return;

	const int val = valIdx + 1;
	const baseType res1 = 1.0 / (baseType)val;
	const baseType res2 = 1.0 / (ARRAY_LENGTH - valIdx);

	sumsCache[0][cacheIdx] = res1;
	sumsCache[1][cacheIdx] = res2;
	if (val % 2) {
		sumsCache[2][cacheIdx] = sumsCache[4][cacheIdx] = 0;
		sumsCache[3][cacheIdx] = res1;
		sumsCache[5][cacheIdx] = res2;
	} else {
		sumsCache[3][cacheIdx] = sumsCache[5][cacheIdx] = 0;
		sumsCache[2][cacheIdx] = res1;
		sumsCache[4][cacheIdx] = res2;
	}

	__syncthreads();
		
	for (int i = blockDim.x / 2; i != 0; i /= 2) {
		if (cacheIdx < i)
			for (int u = 0; u < sumsAmount; u++)
				sumsCache[u][cacheIdx] += sumsCache[u][cacheIdx + i];
		__syncthreads();
	}
	
	if (threadIdx.x) return;
	A[blockIdx.x] = sumsCache[0][0];
	B[blockIdx.x] = sumsCache[1][0];
	C1[blockIdx.x] = sumsCache[2][0];
	C2[blockIdx.x] = sumsCache[3][0];
	D1[blockIdx.x] = sumsCache[4][0];
	D2[blockIdx.x] = sumsCache[5][0];
}

__host__ void showResult(baseType* res) {
	const char* resStr = "A = %.30f\nB = %.30f\nC = C1 + C2 = %.30f + %.30f\n = %.30f\nD = D1 + D2 = %.30f + %.30f\n = %.30f\n";
	printf(resStr, res[0], res[1], res[2], res[3], res[2] + res[3], res[4], res[5], res[4] + res[5]);
}

int main() {
	const int sumsAmount = 6;
	const int sumsSize = BLOCKS * sizeof(baseType);
	baseType** Ah = new baseType*[sumsAmount];
	baseType** Ad = new baseType*[sumsAmount];
	baseType results[sumsAmount];

	auto forEach = [&](lambdaIterator &fn) -> void {
		for (int i = 0; i < sumsAmount; i++) fn(i);
	};

	forEach((lambdaIterator) [&](int i) {
		Ah[i] = new baseType[BLOCKS];
		Ad[i] = nullptr;
		cudaMalloc(&Ad[i], sumsSize);
	});

	calculateSums<<<BLOCKS, THREADS_PER_BLOCK>>>(Ad[0], Ad[1], Ad[2], Ad[3], Ad[4], Ad[5]);

	forEach((lambdaIterator)[&](int i) {
		cudaMemcpy(Ah[i], Ad[i], sumsSize, cudaMemcpyDeviceToHost);
		baseType sum = 0.0;
		for (int u = 0; u < BLOCKS; u++) sum += Ah[i][u];
		results[i] = sum;
	});
		
	showResult(results);

	forEach((lambdaIterator)[&](int i) {
		cudaFree(Ad[i]);
		delete Ah[i];
	});

	delete[] Ah;

	return 0;
}