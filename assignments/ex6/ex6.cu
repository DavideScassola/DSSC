#include <stdio.h>
#include <math.h>

#define N 8192
#define THREAD_PER_BLOCK_SIDE 2

/*
__global__ void multiply(int * in1, int * in2, int * out, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    int startrow = (index / size) * size;
    int startcol = index % size;
    int i;
    int sum = 0;
    for(i = 0; i < size; ++i)
        sum += in1[startrow + i] * in2[startcol + i * size];

    out[index] = sum;
}
*/

__global__ void transpose(int * in, int * out, int size)
{

}

int main()
{

	int * h_in, * h_out;
    int * d_in, * d_out;
    int size = N*N;
    int size_in_memory = size * sizeof(int);
    int i;


	// timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop)


	//allocate memory in host and device
	h_in = (int *)malloc(size_in_memory);
	h_out = (int *)malloc(size_in_memory);

	cudaMalloc((void**)&d_in, size_in_memory);
    cudaMalloc((void**)&d_out, size_in_memory);


	//fill matrix in host
    for(i = 0; i<size; i++)
		in1[i] = i;


	//transfer matrix from host to device
	cudaMemcpy(d_in, h_in, size_in_memory, cudaMemcpyHostToDevice);


	//transpose matrix in device
    dim3 grid, block;
    block.x = THREAD_PER_BLOCK_SIDE;
    block.y = THREAD_PER_BLOCK_SIDE;
    grid.x = N / block.x;
    grid.y = N / block.y;

	cudaEventRecord(start);
    transpose<<< grid, block >>>(d_in, d_out, N);
	cudaEventRecord(stop);


	//transfer matrix in host
	cudaMemcpy(h_out, d_out, size_in_memory, cudaMemcpyDeviceToHost);


	//free memory   
    free(h_in);
	free(h_out);
    cudaFree(d_in);
	cudaFree(d_out);


	//showing Bandwidth
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("Effective Bandwidth (GB/s): %fn", size_in_memory/milliseconds/1e6);


    return 0;
}
