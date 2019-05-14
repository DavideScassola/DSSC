#include <stdio.h>

#define N 8192
#define THREAD_PER_BLOCK_SIDE_X 8
#define THREAD_PER_BLOCK_SIDE_Y 8
#define THREAD_PER_BLOCK THREAD_PER_BLOCK_SIDE_X*THREAD_PER_BLOCK_SIDE_Y
#define TYPE double
#define TYPE_S "double"

__global__ void transpose(TYPE * in, TYPE * out, int size)
{
    //int temp_side = THREAD_PER_BLOCK;
    __shared__ TYPE temp_matrix[THREAD_PER_BLOCK_SIDE_X][THREAD_PER_BLOCK_SIDE_Y];

    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    // copy submatrix (transposed) in shared memory
    temp_matrix[threadIdx.x][threadIdx.y] = in[row*size + col];

    __syncthreads();

    // copy submatrix in main memory
    out[col*size + row] = temp_matrix[threadIdx.x][threadIdx.y];

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
    block.x = THREAD_PER_BLOCK_SIDE_X;
    block.y = THREAD_PER_BLOCK_SIDE_Y;
    grid.x = N / block.x;
    grid.y = N / block.y;

    cudaEventRecord(start);
    transpose<<< grid, block >>>(d_in, d_out, N);
    cudaEventRecord(stop);


    // transfer matrix from device to host
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

    printf("\nmatrix type: %s", TYPE_S);
    printf("\nblock: %d x %d", block.y, block.x);
    printf("\nmilliseconds: %f", milliseconds);
    printf("\nBandwidth: %f GB/s \n", 2*size_in_memory/milliseconds/1e6);


    return 0;
}
