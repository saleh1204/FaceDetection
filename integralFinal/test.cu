#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"

#define BLOCKWIDTH 1
#define NUMTHREAD 5
#define ASIZE 5

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


__global__ void prefixSumRow(int *input, int * output){
	  
	__shared__ int sh_img[ASIZE*ASIZE];
	__shared__ int sh_integral[ASIZE*ASIZE]; // allocated on invocation  
	int thid = blockDim.x * blockIdx.x + threadIdx.x;
	

	int i,j;

	for(i = 0; i<ASIZE;i++)
		sh_img[thid+i] = input[thid+i];


	__syncthreads();  //ensure all the writes to shared memory

	if(thid == 0)	sh_integral[thid] = sh_img[thid];

	
	for (i = 1; i < ASIZE; ++i)
	{
		sh_integral[thid + i] = sh_img[thid+i] + sh_integral[thid-i];
	}

	__syncthreads();


	//Write output
	for(i = 0; i<ASIZE;i++)
		output[thid+i] = sh_integral[thid+i];

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
		srand(i);
		for (j = 0; j < ASIZE; ++j)
		{
			srand(j);
			IMG_HOST[i*ASIZE + j] = rand() % 100; 
		}
	}
	
	printArray(IMG_HOST);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMemcpy(IMG_DEV, IMG_HOST, size*size, cudaMemcpyHostToDevice);
	//cudaMemcpy(INTG_DEV, INTG_HOST, size*size, cudaMemcpyHostToDevice);


	cudaEventRecord(start, 0);

	prefixSumRow <<< NUMTHREAD/BLOCKWIDTH, BLOCKWIDTH >>> (IMG_DEV, INTG_DEV);

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