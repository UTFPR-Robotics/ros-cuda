#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 512

// Global function means it will be executed on the device (GPU)
__global__ void add(int *in1, int *in2, int *out)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	out[index] = in1[index] + in2[index];
}

void random_ints(int *i, int size)
{
	for(int k=0; k<size; k++)
	{
		i[k]=rand()%50;
	}
}

int *testmain(int num) 
{
	int *in1, *in2, *out; // host copies of inputs and output
	int *d_in1, *d_in2, *d_out; // device copies of inputs and output
	int size = num * sizeof(int);

	// Alloc space for device copies of three vectors
	cudaMalloc((void **)&d_in1, size);
	cudaMalloc((void **)&d_in2, size);
	cudaMalloc((void **)&d_out, size);

	// Alloc space for host copies of the three vectors and setup input values
	in1 = (int *)malloc(size); random_ints(in1, num);
	in2 = (int *)malloc(size); random_ints(in2, num);
	out = (int *)malloc(size);

	// Copy inputs to device
	cudaMemcpy(d_in1, in1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in2, in2, size, cudaMemcpyHostToDevice);

	// Launch add() kernel on GPU
	add<<<num/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_in1, d_in2, d_out);
	// Wait for the GPU to finish
	cudaDeviceSynchronize();
	// Copy result back to host
	cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

	// Cleanup
	free(in1); free(in2); free(out);
	cudaFree(d_in1); cudaFree(d_in2); cudaFree(d_out);
	return out;
}
