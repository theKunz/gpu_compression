#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#define N 32

__global__ void add(int* a, int* b, int* c) {
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x]; //each block invocation refers to itself with blockIdx.x
}

void random_ints(int* a, int n);

int main(void) {
	int *a,*b,*c; //cpu copies
	int *gpu_a, *gpu_b, *gpu_c; //gpu copies
	int size = N * sizeof(int);

	//Allocate space on the gpu for a,b,c
	cudaMalloc((void**)&gpu_a, size);
	cudaMalloc((void**)&gpu_b, size);
	cudaMalloc((void**)&gpu_c, size);
	
	a = (int*)malloc(size); random_ints(a, N);
	b = (int*)malloc(size); random_ints(b, N);
	c = (int*)malloc(size);

	//copy values in cpu memory to gpu memory
	cudaMemcpy(gpu_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_b, b, size, cudaMemcpyHostToDevice);

	//run the function
	add<<<N, 1>>>(gpu_a, gpu_b, gpu_c);

	//copy result back to host
	cudaMemcpy(c, gpu_c, size, cudaMemcpyDeviceToHost);

	int i;
	printf("Sum is: ");
	for (i = 0; i < N; i++) {
		printf("%i, ", c[i]);		
	}
	printf("\n");

	cudaFree(gpu_a); cudaFree(gpu_b); cudaFree(gpu_c);
	free(a); free(b); free(c);

	return 0;
}

void random_ints(int* a, int n) {
	int i;
	for (i = 0; i < n; i++) {
		*(a + i) = i;
	}
}
