#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 128

__global__ void av3(int n, float *in1, float *in2, float *in3, float *out)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// Guarantees that index does not go beyond vector size and applies average
	if (index<n)
	{
			out[index] = (in1[index] + in2[index] + in3[index])/3;
	}
}

float *average3(int num, float *in1, float *in2, float *in3, float *out) 
{
	// Device copies of three inputs and output, size of allocated memory, num of threads and blocks
	float *d_in1, *d_in2, *d_in3, *d_out; 
	int size = num * sizeof(float);
	int thr, blk;
	// Alloc memory for device copies of inputs and outputs
	cudaMalloc((void **)&d_in1, size);
	cudaMalloc((void **)&d_in2, size);
	cudaMalloc((void **)&d_in3, size);
	cudaMalloc((void **)&d_out, size);
	// Copy inputs to device
	cudaMemcpy(d_in1, in1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in2, in2, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in3, in3, size, cudaMemcpyHostToDevice);
	// Calculates blocks and threads and launch average3 kernel on GPU
	blk=floor(num/THREADS_PER_BLOCK)+1;
	thr=THREADS_PER_BLOCK;
	av3<<<blk,thr>>>(num, d_in1, d_in2, d_in3, d_out);
	// Wait for the GPU to finish
	cudaDeviceSynchronize();
	// Copy result back to host and cleanup
	cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
	cudaFree(d_in1); cudaFree(d_in2); cudaFree(d_in3); cudaFree(d_out);
	return out;
}
