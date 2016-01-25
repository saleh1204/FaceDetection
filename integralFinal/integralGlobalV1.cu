#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"

#define NUMBLOCK 1
#define BLOCKWIDTH 10
#define NUMTHREAD 1000
#define ASIZE 1000

void printArray(int * image){
	int i,j;
	for (i = 0; i < ASIZE; ++i)
	{
		for (j = 0; j < ASIZE; ++j)
		{
			printf("%d\t", image[i * ASIZE + j]);
		}
		printf("\n");
	}
	printf("\n\n");
}

__global__ void rowSum(int * img, int * integral)
{
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int i;


	printf("blockIdx = %d, blockDim = %d, threadIdx = %d, img[%d] = %d\n", blockIdx.x, blockDim.x, threadIdx.x, idx, img[idx]);

	//integral[idx] = img[idx];

	for(i = 0; i<ASIZE; i++){
		
		integral[idx*ASIZE+i] = img[idx*ASIZE+i] + img[idx*ASIZE+i+1];
		
		__syncthreads();
	}
	
	integral[idx*ASIZE+ASIZE-1] = img[idx*ASIZE+ASIZE-1];
}

__global__ void columnSum(int * img, int * integral)
{
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int i;

	//printf("blockIdx = %d, blockDim = %d, threadIdx = %d, img[%d] = %d\n", blockIdx.x, blockDim.x, threadIdx.x, idx, img[idx]);
	//printf("blockIdy = %d, blockDim = %d, threadIdy = %d, img[%d] = %d\n", blockIdx.y, blockDim.y, threadIdx.y, idy, img[idy]);

	//integral[idx] = img[idx];

	for(i = 0; i<ASIZE; i++){
		integral[idx + i*ASIZE] = img[idx+ (i*ASIZE)] + img[idx+ (i+1)*ASIZE];
		__syncthreads();
	}
	
	integral[idx+(ASIZE)*(ASIZE-1)] = img[idx+(ASIZE)*(ASIZE-1)];
}




int main()
{
	//int ASIZE = *(int *) argv[1];
	int *IMG_HOST, *INTG_HOST;
	int *IMG_DEV, *INTG_DEV;

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
		//srand(i);
		for (j = 0; j < ASIZE; ++j)
		{
			//srand(j);
			IMG_HOST[i*ASIZE + j] = i*2 + j*4;
		}
	}
	
	//printArray(IMG_HOST);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMemcpy(IMG_DEV, IMG_HOST, size*size, cudaMemcpyHostToDevice);


	cudaEventRecord(start, 0);

	rowSum <<< NUMTHREAD/BLOCKWIDTH, BLOCKWIDTH >>> (IMG_DEV, INTG_DEV);

	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&timePassed, start,stop);
	printf("Time Spent Row: %0.5f\n", timePassed);

	cudaMemcpy(IMG_DEV, IMG_HOST, size*size, cudaMemcpyHostToDevice);


	cudaEventRecord(start, 0);

	columnSum <<< NUMTHREAD/BLOCKWIDTH, BLOCKWIDTH >>> (IMG_DEV, INTG_DEV);

	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&timePassed, start,stop);
	printf("Time Spent Row: %0.5f\n", timePassed);

	cudaMemcpy(INTG_HOST, INTG_DEV, size*size, cudaMemcpyDeviceToHost);

	//printArray(INTG_HOST);

	//Free up the resources
	free(IMG_HOST);
	free(INTG_HOST);
	cudaFree(IMG_DEV);
	cudaFree(INTG_DEV);

	return 0;
}
