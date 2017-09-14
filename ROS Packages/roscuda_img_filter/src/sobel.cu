#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define THREADS_PER_BLOCK 128

__global__ void mask(double num, uchar *a, uchar *b, int n)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int aux1,aux2;
	// Guarantees that index does not go beyond vector size and applies filter
	if (index>(n+1) && index<(num-(n+1)))
	{
			 
			 aux1 = (a[index-n-1] + a[index-n]*2 -  a[index+n-1] + a[index-n+1] - a[index+n]*2 - a[index+n+1]);
			 aux2 = (a[index-n-1] + a[index-1]*2 +  a[index+n-1] - a[index-n+1] - a[index+1]*2 - a[index+n+1]);
			 b[index] = abs(aux2-aux1);

			 if(b[index]>255){b[index]=255;}
			 if(b[index]<0){b[index]=0;}

	}else{if(index<num){b[index]=0;}}
}

int *difffilter(cv::Mat src, cv::Mat dst) 
{
	int m = src.rows;
	int n = src.cols;
	double num=n*m;

	std::vector<uchar> v,w;
    v = src.reshape(1,num);
    w.reserve(num);

    //printf("%d\n",v.end()-v.begin());
    //printf("%i %i %i %i %i\n",v[0],v[1],v[2],v[3],v[4]);

	// Device copies of a, b, c, d, size of allocated memory, num of threads and blocks
	uchar *d_a, *d_b; 
	double size = num * sizeof(uchar);
	int thr, blk;
	// Alloc memory for device copies of a, b, c, d
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	// Copy inputs to device
	cudaMemcpy(d_a, v.data(), size, cudaMemcpyHostToDevice);
	// Calculates blocks and threads and launch average3 kernel on GPU
	blk=floor(num/THREADS_PER_BLOCK)+1;
	thr=THREADS_PER_BLOCK;
	mask<<<blk,thr>>>(num, d_a, d_b, n);
	// Wait for the GPU to finish
	cudaDeviceSynchronize();
	// Copy result back to host and cleanup
	cudaMemcpy(w.data(), d_b, size, cudaMemcpyDeviceToHost);
	memcpy(dst.data, w.data(), size);
	cudaFree(d_a); cudaFree(d_b);
	return 0;
}