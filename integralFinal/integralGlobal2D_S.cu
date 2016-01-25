#include <stdlib.h>

#include <time.h>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"

#define ROWSIZE 8192 // Number of Columns
#define COLSIZE 8192 // Number of Rows
#define SIZE (ROWSIZE * COLSIZE) // total Size
#define BLOCKWORK 2
 
#define num_threads 32 // number of threads  per block 
int num_blocks = SIZE/(num_threads*BLOCKWORK) + (SIZE%(num_threads*BLOCKWORK) == 0 ? 0:1); // total number of blocks

//#define NUMBLOCK 1
//#define BLOCKWIDTH 16  
//#define NUMTHREAD 4
//#define ASIZE 4

void printArray(int * image){
	int i,j;
	for (i = 0; i < COLSIZE; ++i)
	{
		for (j = 0; j < ROWSIZE; ++j)
		{
			printf("%d\t", image[i * ROWSIZE + j]);
		}
		printf("\n");
	}
	printf("\n\n");
}

__global__ void prefixSum(int * img, int * integral)
{
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	int index;
	for (int i=0; i<ROWSIZE; i++)
	{
		index = idx + (i*num_threads*BLOCKWORK);
		if (index >= SIZE)break;
		if (index % ROWSIZE == 0)
		{
			integral[index] = img[index];
		}
		else
		{
				
			integral[index] = img[index] + img[index-1];
		}
		i = index;
		__syncthreads();
	}
	
	/*
	if (idx < SIZE)
	{
		if (idx % ROWSIZE == 0)
		{
			integral[idx] = img[idx];
		}
		else
		{
			
			
			integral[idx] = img[idx] + integral[idx-1];
		}
	}
	*/
	//__syncthreads();
	
}

__global__ void columnSum(int * img, int * integral)
{
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	
	
	int index;
	for (int i=0; i<COLSIZE; i++)
	{
		index = idx + (i*num_threads*BLOCKWORK);
		if (index >= SIZE)break;
		if (index== 0)
		{
			integral[index] = img[index];
		}
		else
		{
			// current pixel (col) = original pixel (col) + current col - 1
			// integral[i][j] = img[i][j] + integral[i-1][j]
			integral[index] = img[index] + img[index - ROWSIZE];
		}
		i = index;
		__syncthreads();
	}
	/*
	if (idx < SIZE)
	{
		if (idx == 0)
		{
			integral[idx] = img[idx];
		}
		else
		{
			// current pixel (col) = original pixel (col) + current col - 1
			// integral[i][j] = img[i][j] + integral[i-1][j]
			integral[idx] = img[idx] + integral[idx - ROWSIZE];
		}
	}
	*/
	//__syncthreads();

}

int main()
{
//	const int SIZE = ASIZE;
	//int ASIZE = *(int *) argv[1];
	int *IMG_HOST, *INTG_HOST;
	int *IMG_DEV, *INTG_DEV;
	//const int SIZE = MXSIZE;
	//Time initialization
	float timePassed;

	size_t size = SIZE*sizeof(int);
	cudaSetDevice(1);

	IMG_HOST = (int *)malloc(size);
	INTG_HOST = (int *)malloc(size);

	cudaMalloc((void **) &IMG_DEV, size);
	cudaMalloc((void **) &INTG_DEV, size);


	int i,j;//, random;
	for (i = 0; i < COLSIZE; ++i)
	{
		//srand(i);
		for (j = 0; j < ROWSIZE; ++j)
		{
			//srand(j);
			IMG_HOST[i*ROWSIZE + j] = i*2 + j*4;
		}
	}
	
	//printArray(IMG_HOST);
//	dim3 grid(NUMBLOCK,NUMBLOCK), block(NUMTHREAD,NUMTHREAD);
	dim3 block(ROWSIZE/(num_threads*BLOCKWORK), 1);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMemcpy(IMG_DEV, IMG_HOST, size, cudaMemcpyHostToDevice);


	cudaEventRecord(start, 0);

	prefixSum <<< num_blocks, num_threads*BLOCKWORK >>> (IMG_DEV, INTG_DEV);

	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);


	cudaEventElapsedTime(&timePassed, start,stop);
	printf("Time Spent Row: %0.5f ms\n", timePassed);


//#################################################################//
	
	cudaMemcpy(INTG_HOST, INTG_DEV, size, cudaMemcpyDeviceToHost);
	
	//printArray(INTG_HOST);

	
	//cudaMemcpy(IMG_DEV, INTG_HOST, size, cudaMemcpyHostToDevice);
	

	//INTG_HOST = (int *)malloc(size*size);


	cudaEventRecord(start, 0);

	columnSum <<< num_blocks, num_threads*BLOCKWORK >>> (IMG_DEV, INTG_DEV);

	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&timePassed, start,stop);
	printf("Time Spent Column: %0.5f ms\n", timePassed);

	cudaMemcpy(INTG_HOST, INTG_DEV, size, cudaMemcpyDeviceToHost);

	//printArray(INTG_HOST);

	//Free up the resources
	free(IMG_HOST);
	free(INTG_HOST);
	cudaFree(IMG_DEV);
	cudaFree(INTG_DEV);

	return 0;
}
