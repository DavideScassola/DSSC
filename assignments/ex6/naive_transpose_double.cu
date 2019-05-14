#include <stdio.h>

#define N 8192
#define TYPE double
#define TYPE_S "double"

__global__ void transpose(TYPE * in, TYPE * out, int size)
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    out[col*size + row] = in[row*size + col];
}

int correct(TYPE* a, TYPE* b, int side)
{   
    int i;
    for(i=0; i<side*side; i++)
        if(a[i]!=b[(i%side)*side + i/side]) return 0;
    return 1;
}

int main(int argc, char* argv[])
{

    TYPE * h_in, * h_out;
    TYPE * d_in, * d_out;
    int size = N*N;
    int size_in_memory = size * sizeof(TYPE);
    int i;

    int by = argc<2 ? 32 : atoi(argv[1]); // default is 32
    int bx = argc<3 ? 32 : atoi(argv[2]); // default is 32


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
    block.x = bx;
    block.y = by;
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
