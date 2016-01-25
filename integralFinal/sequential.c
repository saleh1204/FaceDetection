#include <sys/time.h>
#include "stdio.h"
#include "time.h"
#include "stdlib.h"

#define SIZE  10000


/* Return 1 if the difference is negative, otherwise 0.  */
int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
    result->tv_sec = diff / 1000000;
    result->tv_usec = diff % 1000000;

    return (diff<0);
}

void printArray(int * image){
	int i,j;
	for (i = 0; i < SIZE; ++i)
	{
		for (j = 0; j < SIZE; ++j)
		{
			printf("%d\t", image[i * SIZE + j]);
		}
		printf("\n");
	}
	printf("\n\n");
}

int main()
{
	
	int *image, *integral;
	struct timeval tvBegin, tvEnd, tvDiff;
	image = (int *)malloc(SIZE*SIZE*sizeof(int));
	integral = (int *)malloc(SIZE*SIZE*sizeof(int));;
	int i, j;

	//Initializing the image with random variables
	for (i = 0; i < SIZE; ++i)
	{
		for (j = 0; j < SIZE; ++j)
		{
			image[i*SIZE + j] = i*2 + j*8;
		}
	}

	//printf("Array Initalized .... \n");
	//printArray(image);
	clock_t begin, end;
	begin = clock();
	gettimeofday(&tvBegin, NULL);
	//Computing the integral
	integral[0] = image[0];
	for (i = 0; i < SIZE; ++i)
	{
		for (j = 0; j < SIZE; ++j)
		{
			if(i == 0)
				integral[i * SIZE + j] = integral[i * SIZE + j - 1] + image[i * SIZE + j];
			else if(j == 0)
				integral[i * SIZE + j] = integral[(i-1) * SIZE + j] + image[i * SIZE + j];
			else
				integral[i * SIZE + j] = integral[i * SIZE + j-1] + integral[(i-1) * SIZE + j] 
											- integral[(i-1) * SIZE + (j-1)] + image[i * SIZE + j];
		}
	}
	end = clock();
	gettimeofday(&tvEnd, NULL);
	
	float time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("time_spent > %f ms\n", time_spent * 100);
	//printf("Integral Image .....\n");
	//printArray(integral);
	
	// diff
    timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
    printf("Time: %06ld microseconds %f ms\n",(tvDiff.tv_usec), (1.0*tvDiff.tv_usec/1000.0));



	return 0;
}

