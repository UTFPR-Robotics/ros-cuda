#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 128

__global__ void av3(int n, float *a, float *b, float *c, float *d)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// Guarantees that index does not go beyond vector size and applies average
	if (index<n)
	{
			d[index] = (a[index] + b[index] + c[index])/3;
	}
}

float *average3(int num, float *a, float *b, float *c, float *d) 
{
	// Device copies of a, b, c, d, size of allocated memory, num of threads and blocks
	float *d_a, *d_b, *d_c, *d_d; 
	int size = num * sizeof(float);
	int thr, blk;
	// Alloc memory for device copies of a, b, c, d
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	cudaMalloc((void **)&d_d, size);
	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);
	// Calculates blocks and threads and launch average3 kernel on GPU
	blk=floor(num/THREADS_PER_BLOCK)+1;
	thr=THREADS_PER_BLOCK;
	av3<<<blk,thr>>>(num, d_a, d_b, d_c, d_d);
	// Wait for the GPU to finish
	cudaDeviceSynchronize();
	// Copy result back to host and cleanup
	cudaMemcpy(d, d_d, size, cudaMemcpyDeviceToHost);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_d);
	return d;
}
