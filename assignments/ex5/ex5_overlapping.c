#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>

void vector_sum(int* out, int* in, size_t size)
{
    size_t i;
    for(i = 0; i<size; i++)
        out[i] += in[i];
}

void fill_vector(int* v, int x, size_t size)
{
    size_t i;
    for(i = 0; i<size; i++)
        v[i] = x;  
}

void print_vector(int* v, size_t size)
{
    size_t i;
    for(i = 0; i<size; i++)
        printf("%d ", v[i]);

    printf("\n");
}

void swapPointers(int** a, int** b)
{
    int* t = *a;
    *a = *b;
    *b = t;
}

int main(int argc, char* argv[])
{
    
    int rank; // identifier of the process
    int np; // total number of MPI processes
    int root = 0; // rank of the "root" process

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &np );
    MPI_Request request;
    MPI_Status status;

    double t1, t2;
    size_t N = argc<2 ? 1011 : atoi(argv[1]); // default is 1011
    int i;
    int* Xr = malloc(sizeof(int)*N);
    int* Xs = malloc(sizeof(int)*N);
    fill_vector(Xs, rank, N);
    int* sum = malloc(sizeof(int)*N);
    fill_vector(sum, 0, N);
    
    t1 = MPI_Wtime();
    for(i=0; i<np; i++)
    {
        MPI_Isend(Xs, N, MPI_INT, (rank+1)%np, 101, MPI_COMM_WORLD, &request);
        vector_sum(sum, Xs, N);
        MPI_Recv(Xr, N, MPI_INT, (rank-1+np)%np, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Wait(&request, &status);
        swapPointers(&Xr,&Xs);
    }
    printf("I'm %d and my sum is %d,...\n", rank, sum[0]);
    
    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    if(rank==root)
        printf("\nelapsed-time: %f\n", t2-t1);
    MPI_Finalize();

    free(Xr);
    free(Xs);
    free(sum);
    
}
