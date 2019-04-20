#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>

void pm(double* m, size_t n)
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
            MPI_Send(partial_matrix, local_size, MPI_DOUBLE, root, 101, MPI_COMM_WORLD);
            //printf("proc[%d] sent\n", rank);
        }
        else
        {
            double* m = (double*) malloc(sizeof(double)*size);
            size_t current_size, partial_size = size/np, starting_point = 0; 
            int i;
            for(i = 0; i<np; i++)
            {
                current_size = partial_size + (i<rest);

                if(i==root)
                    copy(m+starting_point, partial_matrix, current_size);
                else
                    MPI_Recv(m+starting_point, current_size, MPI_DOUBLE, i, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                starting_point+=current_size;
            }
            pm(m,N);
            free(m);    
        }
    }
    else
    {
        if(rank!=root)
        {
            MPI_Send(partial_matrix, local_size, MPI_DOUBLE, root, 101, MPI_COMM_WORLD);
        }
        else
        {
            MPI_Request request;
            MPI_Status status;
            FILE* data_file;
            data_file=fopen("ex4_data.dat","wb");

            size_t max_chunk_size = (size/np + 1);
            double* buffer1 = partial_matrix;
            double* buffer2 = (double*) malloc(sizeof(double)*max_chunk_size);
            int i;
            double* receiving_buffer;
            double* writing_buffer;
            size_t receiving_current_size, writing_current_size, partial_size = size/np;

            writing_current_size = partial_size + (0<rest);
            receiving_current_size = partial_size + (1<rest);

            receiving_buffer = buffer2;
            MPI_Irecv(receiving_buffer, receiving_current_size, MPI_DOUBLE, 1, 101, MPI_COMM_WORLD, &request);
            fwrite(partial_matrix, sizeof(double), writing_current_size, data_file);
            MPI_Wait(&request, &status);

            for(i = 1; i<np-1; i++)
            {
                receiving_current_size = partial_size + (i+1<rest);
                writing_current_size = partial_size + (i<rest);

                receiving_buffer = (i%2 == 1) ? buffer1 : buffer2;
                writing_buffer = (i%2 == 1) ? buffer2 : buffer1;

                MPI_Irecv(receiving_buffer, receiving_current_size, MPI_DOUBLE, i+1, 101, MPI_COMM_WORLD, &request);
                fwrite(writing_buffer, sizeof(double), writing_current_size, data_file);
                MPI_Wait(&request, &status);

                //MPI_Irecv(buffer, current_size, MPI_DOUBLE, 1, 101, MPI_COMM_WORLD, &request);
                //fwrite(buffer, sizeof(double), current_size, data_file);
                //printf("ricevuto il messaggio di:%d, current_size=%ld\n", i, current_size);
                
                //printf("scrittura completata(%d)\n", i);

                //fseek(data_file, 0, SEEK_CUR);
                //printf("testina avanti di %ld double \n", (current_size-N*N/total_np));
            }
            writing_current_size = partial_size + (np-1<rest);
            fwrite(receiving_buffer, sizeof(double), writing_current_size, data_file);

            free(buffer2);
            fclose(data_file);

            // verify correctness
            double* complete_mat = (double*) malloc(sizeof(double)*N*N);
            data_file = fopen("ex4_data.dat","r");
            fread(complete_mat,sizeof(double),N*N,data_file);
            pm(complete_mat,N);
            free(complete_mat);
            fclose(data_file);   
        }
    }

    free(partial_matrix);
    MPI_Finalize();
}