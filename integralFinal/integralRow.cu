#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"

#define BLOCKWIDTH 1
#define NUMTHREAD 100
#define ASIZE 100

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
	__shared__ int sh_img[ASIZE*ASIZE];
	__shared__ int sh_integral[ASIZE*ASIZE];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int i;



	for(i=0;i<ASIZE;i++){
		sh_img[idx*ASIZE + i] = img[idx*ASIZE + i];
	}

	__syncthreads();

	sh_integral[idx*ASIZE] = sh_img[idx*ASIZE];

	int address = idx*ASIZE;
	for(i = 1;i<ASIZE;i++){
		sh_integral[address+i] = sh_img[address+i] + sh_integral[address-i];
	}

	__syncthreads();

	for(i=0;i<ASIZE;i++)
		integral[idx*ASIZE+i] = sh_integral[idx*ASIZE+i];

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
	
	printArray(IMG_HOST);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMemcpy(IMG_DEV, IMG_HOST, size*size, cudaMemcpyHostToDevice);


	cudaEventRecord(start, 0);

	prefixSum <<< NUMTHREAD/BLOCKWIDTH, BLOCKWIDTH >>> (IMG_DEV, INTG_DEV);

	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&timePassed, start,stop);
	printf("Time Spent: %0.5f\n", timePassed);

	cudaMemcpy(INTG_HOST, INTG_DEV, size*size, cudaMemcpyDeviceToHost);

	printArray(INTG_HOST);

	//Free up the resources
	free(IMG_HOST);
	free(INTG_HOST);
	cudaFree(IMG_DEV);
	cudaFree(INTG_DEV);

	return 0;
}