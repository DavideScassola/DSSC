#include <stdio.h>
#include <math.h>

#define N 8
#define THREAD_PER_BLOCK 2

__global__ void multiply(int * in1, int * in2, int * out, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int startrow = (index / size) * size;
    int startcol = index % size;
    int i;
    int sum = 0;
    for(i = 0; i < size; ++i) {
        sum += in1[startrow + i] * in2[startcol + i * size];
    }
    out[index] = sum;
}

int main()
{
    int * in1, * in2, * out;
    int * d_in1, * d_in2, * d_out;
    int size = N * N * sizeof(int);
    int i;

    cudaMalloc((void**)&d_in1, size);
    cudaMalloc((void**)&d_in2, size);
    cudaMalloc((void**)&d_out, size);
    
    in1 = (int *)malloc(size);
    in2 = (int *)malloc(size);
    out = (int *)malloc(size);
    
    for(i = 0; i<N*N; ++i)
    {
	in1[i] = i%N;
        in2[i] = i%N -1;	
    } 
    
    cudaMemcpy(d_in1, in1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2, in2, size, cudaMemcpyHostToDevice);
    
    multiply<<< N*N/THREAD_PER_BLOCK, THREAD_PER_BLOCK >>>(d_in1, d_in2,  d_out, N);
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    for(i=0; i<N*N; ++i)
    {
	printf("%2d  ", in1[i]);
        if((i+1)%N == 0) {
	    printf("\n");
	}
    }
    printf("--------\n");
    for(i=0; i<N*N; ++i)
    {
        printf("%2d  ", in2[i]);
        if((i+1)%N == 0) {
            printf("\n");
        }
    }
    printf("--------\n");
    for(i=0; i<N*N; ++i)
    {
        printf("%2d  ", out[i]);
        if((i+1)%N == 0) {
            printf("\n");
        }
    }
    free(in1); free(in2); free(out);
    cudaFree(d_in1); cudaFree(d_in2);
    cudaFree(d_out);
    return 0;
}
