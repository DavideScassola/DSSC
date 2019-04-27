#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>

void copy(int* out, int* in, size_t size)
{
    size_t i;
    for(i = 0; i<size; i++)
        out[i] = in[i];
}

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
    size_t N = argc<2 ? 20 : atoi(argv[1]); // default is 20
    int i;
    int* X = malloc(sizeof(int)*N);
    fill_vector(X, rank, N);
    int* sum = malloc(sizeof(int)*N);
    //copy(sum, X, N);
    
    fill_vector(sum, 0, N);

    t1 = MPI_Wtime();
    for(i=0; i<np; i++)
    {
        
        //MPI_Send(X, N, MPI_INT, (rank+1)%np,    101, MPI_COMM_WORLD);
        //printf("%d start sending %d\n", rank, X[0]);
        MPI_Isend(X, N, MPI_INT, (rank+1)%np, 101, MPI_COMM_WORLD, &request);
        vector_sum(sum, X, N);
        MPI_Wait(&request, &status);
        //printf("%d sent %d (ready to receive)\n", rank, X[0]);
        MPI_Recv(X, N, MPI_INT, (rank-1+np)%np, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //MPI_Wait(&request, &status);
        //printf("%d received %d\n", rank, X[0]);
        
    }
    printf("I'm %d and my sum is ", rank);
    print_vector(sum, 4);
    
    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    if(rank==root)
        printf("\nelapsed-time: %f\n", t2-t1);
    MPI_Finalize();

    /*
    size_t N = argc<2 ? 20 : atoi(argv[1]); // default is 20
    int i;
    int* X = malloc(sizeof(int)*N);
    fill_vector(X, rank, N);
    int* sum = malloc(sizeof(int)*N);;
    copy(sum, X, N);

    for(i=0; i<np-1; i++)
    {
        MPI_Send(X, N, MPI_INT, (rank+1)%np,    101, MPI_COMM_WORLD);
        MPI_Recv(X, N, MPI_INT, (rank-1+np)%np, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        vector_sum(sum, X, N);
    }

    printf("I'm %d and my sum is ", rank);
    print_vector(sum, N);

    MPI_Finalize();
    */


    /*
    int i;
    int X = rank;
    int sum = X;
    for(i=0; i<np-1; i++)
    {
        MPI_Send(&X, 1, MPI_INT, (rank+1)%np,    101, MPI_COMM_WORLD);
        MPI_Recv(&X, 1, MPI_INT, (rank-1+np)%np, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        sum+=X;
    }

    printf("I'm %d and my sum is %d\n", rank, sum);

    MPI_Finalize();
    */

}
