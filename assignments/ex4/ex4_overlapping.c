#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#define FILE_NAME

void print_matrix(double* m, size_t n)
{
    size_t i, j;
    for(i=0; i<n; i++)
    {
        for(j=0; j<n; j++)
            printf("%.0f ", m[i*n+j]);
        printf("\n");
    }
}

void copy(double* out, double* in, size_t size)
{
    size_t i;
    for(i = 0; i<size; i++)
        out[i] = in[i];
}

size_t local2global(size_t local_i, size_t rank, size_t total_size, int np)
{
    size_t rest = total_size%np;
    size_t offset_to_add = rank<rest ? rank : rest;
    return (total_size/np)*rank + offset_to_add  + local_i;
}

short int is_on_diagonal(size_t i, size_t dim)
{
    return i%dim == i/dim;
}

void swapBuffers(double** a, double** b)
{
    double* t = *a;
    *a = *b;
    *b = t;
}

int main(int argc, char* argv[])
{
    
    int rank; // identifier of the process
    int np; // total number of MPI processes
    int root = 0; // rank of the "root" process

    size_t N = argc<2 ? 9 : atoi(argv[1]); // default is 9
    size_t size = N*N;

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &np );

    if(np==1)
    {
        printf("please spawn at least 2 processes\n");
        MPI_Finalize();
        return 0;
    }

    size_t rest = size%np;
    short int has_rest = rank<rest;
    size_t local_size = size/np+has_rest;
    double* partial_matrix = (double*) malloc(local_size * sizeof(double));

    size_t global_i;
    size_t local_i;

    for(local_i = 0; local_i<local_size; local_i++)
    {
        //double marker = (double) rank/10;
        global_i = local2global(local_i, rank, size, np);
        partial_matrix[local_i] = is_on_diagonal(global_i, N) /*+ marker*/;
    }


    
    if(N<10)
    {
        if(rank!=root)
        {
            // every process sends to the root its partial matrix
            MPI_Send(partial_matrix, local_size, MPI_DOUBLE, root, 101, MPI_COMM_WORLD);
        }
        else
        {
            double* m = (double*) malloc(sizeof(double)*size); // this will host the complete matrix
            size_t current_size, partial_size = size/np, starting_point = 0; 
            int i;

            for(i = 0; i<np; i++)
            {
                current_size = partial_size + (i<rest);

                // First, I just copy the local partial matrix in the complete matrix
                if(i==root)
                    copy(m+starting_point, partial_matrix, current_size);  

                // Then, I place in the complete matrix all the other partial matrices (receiving them in an ordered way)
                else
                    MPI_Recv(m+starting_point, current_size, MPI_DOUBLE, i, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                starting_point+=current_size;
            }
            print_matrix(m,N); // showing the complete matrix
            free(m);    
        }
    }
    else
    {
        if(rank!=root)
        {
            // every process send to the root its partial matrix
            MPI_Send(partial_matrix, local_size, MPI_DOUBLE, root, 101, MPI_COMM_WORLD);
        }
        else
        {
            MPI_Request request;
            MPI_Status status;
            FILE* data_file;
            data_file=fopen(FILE_NAME,"wb");

            size_t max_chunk_size = (size/np + 1);

            // I will use two buffers in order to overlap communication and I/O
            double* buffer1 = partial_matrix;
            double* buffer2 = (double*) malloc(sizeof(double)*max_chunk_size);
            double* receiving_buffer; // This buffer will hold the partial matrix to receive from the next process
            double* writing_buffer; // This buffer will hold the partial matrix to write down in a file
            
            size_t receiving_current_size, writing_current_size, partial_size = size/np;

            /*
            During the first iteration I receive the partial matrix from process 1
            while I write the partial matrix of the root process.
            */
            writing_current_size = partial_size + (0<rest);
            receiving_current_size = partial_size + (1<rest);

            receiving_buffer = buffer2;
            writing_buffer = buffer1;

            MPI_Irecv(receiving_buffer, receiving_current_size, MPI_DOUBLE, 1, 101, MPI_COMM_WORLD, &request);
            fwrite(writing_buffer, sizeof(double), writing_current_size, data_file);
            MPI_Wait(&request, &status);

            int i;
            // In this cycle I write down the partial matrix from process i while receiving the one from i+1
            for(i = 1; i<np-1; i++)
            {
                receiving_current_size = partial_size + (i+1<rest);
                writing_current_size = partial_size + (i<rest);

		swapBuffers(&receiving_buffer, &writing_buffer);

                MPI_Irecv(receiving_buffer, receiving_current_size, MPI_DOUBLE, i+1, 101, MPI_COMM_WORLD, &request);
                fwrite(writing_buffer, sizeof(double), writing_current_size, data_file);
                MPI_Wait(&request, &status);
            }

            // during the last iteration I just write down the last partial matrix
            writing_current_size = partial_size + (np-1<rest);
            fwrite(receiving_buffer, sizeof(double), writing_current_size, data_file);

            free(buffer2);
            fclose(data_file);

            // Now I read the file for verifying the correctness
            double* complete_mat = (double*) malloc(sizeof(double)*N*N);
            data_file = fopen(FILE_NAME,"r");
            fread(complete_mat,sizeof(double),N*N,data_file);
            print_matrix(complete_mat,N);

            free(complete_mat);
            fclose(data_file);   
        }
    }

    free(partial_matrix);
    MPI_Finalize();
}
