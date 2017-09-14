#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kernel()
{

}

void cudamain() 
{	
	kernel<<<1,1>>>();
	return;
}
