#include <cuda_profiler_api.h>
#include <stdio.h>

int main()
{
    cudaError_t cerr;
    cerr = cudaProfilerStart();
    if (cerr != cudaSuccess) {
        fprintf(stdout, "Error with cudeProfilerStart with error %s\n", cudaGetErrorString(cerr));
    }
    const unsigned int N = 1048576;
    const unsigned int bytes = N * sizeof(int);
    int *h_a = (int*)malloc(bytes);
    int *d_a;
    cerr = cudaMalloc((int**)&d_a, bytes);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "Error with cudaMalloc with error %s\n", cudaGetErrorString(cerr));
    }

    memset(h_a, 0, bytes);
    cerr = cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "Error with Host to Device cudaMemcpy with error %s\n", cudaGetErrorString(cerr));
    }
    cerr = cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "Error with Device to Host cudaMemcpy with error %s\n", cudaGetErrorString(cerr));
    }
    cerr = cudaFree(d_a);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "Error with cudaFree with error %s\n", cudaGetErrorString(cerr));
    }
    cerr = cudaProfilerStop();
    if (cerr != cudaSuccess) {
        fprintf(stderr, "Error with cudaProfileStop with error %s\n", cudaGetErrorString(cerr));
    }
    return 0;
}
