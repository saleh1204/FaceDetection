#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"

#define NUMBLOCK 1
#define BLOCKWIDTH 16  
#define NUMTHREAD 4
#define ASIZE 4

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

__global__ void prefixSum(int * img, int * integral)
{
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int i;

	printf("blockIdx = %d, blockDim = %d, threadIdx = %d, img[%d] = %d\n", blockIdx.x, blockDim.x, threadIdx.x, idx, img[idx]);
	printf("blockIdy = %d, blockDimy = %d, threadIdy = %d, img[%d][%d] = %d\n", blockIdx.y, blockDim.y, threadIdx.y, idx,idy, img[idx*ASIZE + idy]);



	//printf("blockIdx = %d, blockDim = %d, threadIdx = %d, img[%d] = %d\n", blockIdx.x, blockDim.x, threadIdx.x, idx, img[idx]);

	if (idy == 0)
	{
		integral[idx*ASIZE+idy] = img[idx*ASIZE+idy];
	}
	else
		integral[idx*ASIZE+idy] = img[idx*ASIZE+idy] + integral[idx*ASIZE+idy-1];

	printf("img[%d][%d] > %d, integral[] > %d\n", idx, idy,img[idx*ASIZE+idy], integral[idx*ASIZE+idy-1]);

	__syncthreads();
	
}

__global__ void columnSum(int * img, int * integral)
{
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	int i;

	printf("idx > %d, idy > %d, img[] > %d, integral[] > %d\n", idx, idy, img[idx + idy*ASIZE], integral[idx + idy*ASIZE]);

	if (idx == 0)
		integral[idx + idy*ASIZE] = img[idx + idy*ASIZE];
	else
		integral[idx + idy*ASIZE] = img[idx + (idy*ASIZE)] + integral[idx + (idy-1)*ASIZE];

	__syncthreads();

}

int main()
{
//	const int SIZE = ASIZE;
	//int ASIZE = *(int *) argv[1];
	int *IMG_HOST, *INTG_HOST;
	int *IMG_DEV, *INTG_DEV;

	//Time initialization
	float timePassed;

	size_t size = ASIZE*sizeof(int);


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
	
	printArray(IMG_HOST);
	dim3 grid(NUMBLOCK,NUMBLOCK), block(NUMTHREAD,NUMTHREAD);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMemcpy(IMG_DEV, IMG_HOST, size*size, cudaMemcpyHostToDevice);


	cudaEventRecord(start, 0);

	prefixSum <<< grid, block >>> (IMG_DEV, INTG_DEV);

	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);


	cudaEventElapsedTime(&timePassed, start,stop);
	printf("Time Spent Row: %0.5f\n", timePassed);


//#################################################################//
	
	cudaMemcpy(INTG_HOST, INTG_DEV, size*size, cudaMemcpyDeviceToHost);
	
	printArray(INTG_HOST);

	
	//cudaMemcpy(INTG_DEV, INTG_HOST, size*size, cudaMemcpyHostToDevice);
	

	//INTG_HOST = (int *)malloc(size*size);


	cudaEventRecord(start, 0);

	columnSum <<< grid, block >>> (INTG_DEV, INTG_DEV);

	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&timePassed, start,stop);
	printf("Time Spent Column: %0.5f\n", timePassed);

	cudaMemcpy(INTG_HOST, INTG_DEV, size*size, cudaMemcpyDeviceToHost);

	printArray(INTG_HOST);

	//Free up the resources
	free(IMG_HOST);
	free(INTG_HOST);
	cudaFree(IMG_DEV);
	cudaFree(INTG_DEV);

	return 0;
}
