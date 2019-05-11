#include <stdio.h>
#include <math.h>

#define N 8192
#define THREAD_PER_BLOCK_SIDE 32
#define THREAD_PER_BLOCK THREAD_PER_BLOCK_SIDE*THREAD_PER_BLOCK_SIDE

__global__ void transpose(int * in, int * out, int size)
{
    int temp_side = THREAD_PER_BLOCK;
    __shared__ int temp_matrix[THREAD_PER_BLOCK_SIDE][THREAD_PER_BLOCK_SIDE];

    //int temp_i = threadIdx.y*temp_side + threadIdx.x;
    //int temp_i_t = threadIdx.x*temp_side + threadIdx.y;
    int global_i = blockIdx.y*blockDim.y*size + blockIdx.x*blockDim.x + threadIdx.y*size + threadIdx.x;
    int global_i_t = blockIdx.x*blockDim.y*size + blockIdx.y*blockDim.x + threadIdx.y*size + threadIdx.x;

    // copy submatrix (transposed) in shared memory
    temp_matrix[threadIdx.x][threadIdx.y] = in[global_i_t];

    //__syncthreads();

    // copy submatrix in main memory
    out[global_i] = temp_matrix[threadIdx.y][threadIdx.x];

}

int correct(int* a, int* b, int size)
{   
    int i;
    for(i=0; i<size; i++)
        if(a[i]!=b[(i%size)*size + i/size]) return 0;
    return 1;
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
    cudaEventCreate(&stop);


    //allocate memory in host and device
    h_in = (int *)malloc(size_in_memory);
    h_out = (int *)malloc(size_in_memory);

    cudaMalloc((void**)&d_in, size_in_memory);
    cudaMalloc((void**)&d_out, size_in_memory);


    //fill matrix in host
    for(i = 0; i<size; i++)
        h_in[i] = i;


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


    // correctness test
    printf("\ncorrecteness: %d \n", correct(d_in, d_out, size));
   
    //showing Bandwidth
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("\nEffective Bandwidth (GB/s): %f \n", size_in_memory/milliseconds/1e6);


    return 0;
}
