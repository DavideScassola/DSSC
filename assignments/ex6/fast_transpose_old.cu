#include <stdio.h>
#include <math.h>

#define N 8192
#define THREAD_PER_BLOCK_SIDE 32
#define THREAD_PER_BLOCK THREAD_PER_BLOCK_SIDE*THREAD_PER_BLOCK_SIDE
#define TYPE double

__global__ void transpose(TYPE * in, TYPE * out, int size)
{
    //int temp_side = THREAD_PER_BLOCK;
    __shared__ TYPE temp_matrix[THREAD_PER_BLOCK_SIDE][THREAD_PER_BLOCK_SIDE];

    //int temp_i = threadIdx.y*temp_side + threadIdx.x;
    //int temp_i_t = threadIdx.x*temp_side + threadIdx.y;
    int global_i = blockIdx.y*blockDim.y*size + blockIdx.x*blockDim.x + threadIdx.y*size + threadIdx.x;
    int global_i_t = blockIdx.x*blockDim.y*size + blockIdx.y*blockDim.x + threadIdx.y*size + threadIdx.x;

    // copy submatrix (transposed) in shared memory
    temp_matrix[threadIdx.x][threadIdx.y] = in[global_i_t];

    __syncthreads();

    // copy submatrix in main memory
    out[global_i] = temp_matrix[threadIdx.y][threadIdx.x];

}

int correct(TYPE* a, TYPE* b, int side)
{   
    int i;
    for(i=0; i<side*side; i++)
        if(a[i]!=b[(i%side)*side + i/side]) return 0;
    return 1;
}

int main()
{

    TYPE * h_in, * h_out;
    TYPE * d_in, * d_out;
    int size = N*N;
    int size_in_memory = size * sizeof(TYPE);
    int i;


    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    //allocate memory in host and device
    h_in = (TYPE *)malloc(size_in_memory);
    h_out = (TYPE *)malloc(size_in_memory);

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


    // correctness test
    printf("\ncorrecteness: %d \n", correct(h_in, h_out, N));
   

    //free memory   
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);


    //showing Bandwidth
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("\nmilliseconds: %f", milliseconds);
    printf("\nBandwidth: %f GB/s \n", size_in_memory/milliseconds/1e6);


    return 0;
}
