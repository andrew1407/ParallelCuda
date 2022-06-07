#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

using namespace std;

// input parameters
#define MATRIX_SIZE 211									// matrix rows and columns
#define MATRIX_LENGTH (MATRIX_SIZE * MATRIX_SIZE)

typedef int BASE_TYPE;

// GPU calculations
__global__ void matrixMult(const BASE_TYPE* matrixA, const BASE_TYPE* matrixB, BASE_TYPE* matrixResult) {
	const int i = MATRIX_SIZE * (blockDim.y * blockIdx.y + threadIdx.y);
	const int j = blockDim.x * blockIdx.x + threadIdx.x;
	const int resIndex = MATRIX_SIZE * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= MATRIX_LENGTH || j >= MATRIX_LENGTH || resIndex >= MATRIX_LENGTH)
		return;
	BASE_TYPE resValue = 0;
	for (int k = 0; k < MATRIX_SIZE; k++)
		resValue += matrixA[i + k] * matrixB[k * MATRIX_SIZE + j];
	matrixResult[resIndex] = resValue;
}

// random values matrix generator/writer
__host__ void generateInputMatrix(const char *filename) {
	ofstream input(filename);
	if (!input.is_open()) return;
	for (int k = 0; k < 2; k++) {
		string matrix = "";
		for (int i = 0; i < MATRIX_LENGTH; i++) {
			const BASE_TYPE randEl = rand() % 1000;
			input << to_string(randEl) + "\n";
		}
		input << "\n";
	}
	input.close();
}

// reading A & B matrix for calculating
__host__ void readInputMatrix(const char* filename, BASE_TYPE *matrixA, BASE_TYPE *matrixB) {
	ifstream input(filename);
	if (!input.is_open()) return;
	string matrixStr;
	int mtxCounter = 0;
	int i = 0;
	while (getline(input, matrixStr)) {
		if (mtxCounter > 1) break;
		if (!matrixStr.length()) {
			mtxCounter++;
			i = 0;
			continue;
		}
		BASE_TYPE* mtx = mtxCounter ? matrixB : matrixA;
		mtx[i++] = stoi(matrixStr);
	}
	input.close();
}

// writing result of multiiplication
__host__ void writeOutputMatrix(const char* filename, const BASE_TYPE* res) {
	ofstream input(filename);
	if (!input.is_open()) return;
	for (int i = 0; i < MATRIX_LENGTH; i++)
		input << to_string(res[i]) + "\n";
	input.close();
}

int main() {
	//input filenames
	string inputf, outputf;
	cout << "Input filename: ";
	cin >> inputf;
	cout << "Output filename: ";
	cin >> outputf;

	// uncomment to generate input file
	generateInputMatrix(inputf.c_str());

	const int allocatedMem = MATRIX_LENGTH * sizeof(BASE_TYPE);

	BASE_TYPE* matrixAh = new BASE_TYPE[MATRIX_LENGTH];
	BASE_TYPE* matrixBh = new BASE_TYPE[MATRIX_LENGTH];
	BASE_TYPE* matrixCh = new BASE_TYPE[MATRIX_LENGTH];

	BASE_TYPE* matrixAd = nullptr;
	BASE_TYPE* matrixBd = nullptr;
	BASE_TYPE* matrixCd = nullptr;

	readInputMatrix(inputf.c_str(), matrixAh, matrixBh);

	cudaMalloc(&matrixAd, allocatedMem);
	cudaMalloc(&matrixBd, allocatedMem);
	cudaMalloc(&matrixCd, allocatedMem);

	cudaMemcpy(matrixAd, matrixAh, allocatedMem, cudaMemcpyHostToDevice);
	cudaMemcpy(matrixBd, matrixBh, allocatedMem, cudaMemcpyHostToDevice);

	auto start = chrono::high_resolution_clock::now();
	matrixMult<<<dim3(MATRIX_SIZE, MATRIX_SIZE), 1>>>(matrixAd, matrixBd, matrixCd);
	auto end = chrono::high_resolution_clock::now();

	cudaMemcpy(matrixCh, matrixCd, allocatedMem, cudaMemcpyDeviceToHost);

	// testing result
	for (int i = 0; i < MATRIX_SIZE; i++)
		for (int j = 0; j < MATRIX_SIZE; j++) {
			BASE_TYPE sum = 0;
			for (int k = 0; k < MATRIX_SIZE; k++)
				sum += matrixAh[i * MATRIX_SIZE + k] * matrixBh[k * MATRIX_SIZE + j];
			if (matrixCh[i * MATRIX_SIZE + j] != sum) {
				cerr << "Result verification failed at element [" << i << ", " << j << "]!\nsum = " << sum << ", " << matrixCh << "[[i * Bcols + j] = " << matrixCh[i * MATRIX_SIZE + j] << endl;
				exit(EXIT_FAILURE);
			}
		}

	writeOutputMatrix(outputf.c_str(), matrixCh);

	delete[] matrixAh;
	delete[] matrixBh;
	delete[] matrixCh;

	cudaFree(matrixAd);
	cudaFree(matrixBd);
	cudaFree(matrixCd);

	// complete time log
	auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
	cout << "Calculated: " << fixed << duration.count() * 1e-6 << " seconds." << endl;

	return 0;
}
