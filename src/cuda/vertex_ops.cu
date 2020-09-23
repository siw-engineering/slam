#include <stdio.h>
#include<stdlib.h> 

__global__ void test(void)
{
	printf("Hello World! from thread [%d,%d] \
		From device\n", threadIdx.x,blockIdx.x); 
}


void test_wrapper(void)
{
	test<<<1,1>>>();
}