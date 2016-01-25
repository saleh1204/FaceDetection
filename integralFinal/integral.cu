#include "stdio.h"
#include "stdlib.h"
#include "cuda.h"
#include "cuda_runtime.h"

#define SIZE 50

__global__ void scan(float *input, float *output, int n){
	  
	extern __shared__ float temp[]; // allocated on invocation  
	int thid = threadIdx.x;
	int offset = 1;

	temp[2*thid] = input[2*thid];
	temp[2*thid+1] = input[2*thid+1];
	

	int d;
	for (d = n>>1; d > 0; d >>=1)
	{
		__syncthreads();
		if(thid < d)
		{
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	 
	if(thid ==0) temp[n - 1] = 0;

	for(d = 1; d < n; d *= 2)
	{
		offset >>= 1;
		__syncthreads();
		if(thid < d)
		{
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	__syncthreads();

	//Write output
	output[2*thid] = temp[2*thid];
	output[2*thid+1] = temp[2*thid+1];

}

int main()
{
	int *IMG_HOST, *INTG_HOST;
	int *INTG_DEV, *INTG_DEV;

	//Time initialization
	float timePassed;

	long size = ASIZE * sizeof(int);


	IMG_HOST = (int *)malloc(size*size);
	INTG_HOST = (int *)malloc(size*size);

	cudaMalloc((void **) &IMG_DEV, size*size);
	cudaMalloc((void **) &INTG_DEV, size*size);


	int i,j, random;
	for (i = 0; i < ASIZE; ++i)
	{
		srand(i);
		for (j = 0; j < ASIZE; ++j)
		{
			srand(j);
			IMG_HOST[i*ASIZE + j] = rand() % 100; 
		}
	}
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMemcpy(IMG_DEV, IMG_HOST, size*size, cudaMemcpyHostToDevice);
	cudaMemcpy(INTG_DEV, INTG_HOST, size*size, cudaMemcpyHostToDevice);


	cudaEventRecord(start, 0);

	scan <<< 3,3 >>> (IMG_DEV, INTG_DEV);

	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&timePassed, start,stop);
	printf("Time Spent: %0.5f\n", timePassed);

	cudaMemcpy(INTG_HOST, INTG_DEV, size*size, cudaMemcpyDeviceToHost);

	//Free up the resources
	free(IMG_HOST);
	free(INTG_HOST);
	cudaFree(IMG_DEV);
	cudaFree(INTG_DEV);

	
	return 0;
}