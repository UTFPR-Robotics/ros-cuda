#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void add(int *a, int *b, int *c)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	c[index] = a[index] + b[index];
}

void testmain(int size, int *c) 
{
	int *a, *b; // host copies of a, b, c
	int *d_a, *d_b, *d_c; // device copies of a, b, c;
	// Alloc space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); a[0]=1;
	b = (int *)malloc(size); b[0]=4;
	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	// Launch add() kernel on GPU
	add<<<1,1>>>(d_a, d_b, d_c);
	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	// Cleanup
	free(a); free(b);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	return;
}
