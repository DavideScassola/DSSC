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

    double t1, t2;
    size_t N = argc<2 ? 20 : atoi(argv[1]); // default is 20
    int i;
    int* X = malloc(sizeof(int)*N);
    fill_vector(X, rank, N);
    int* sum = malloc(sizeof(int)*N);;
    copy(sum, X, N);

    t1 = MPI_Wtime();
    for(i=0; i<np-1; i++)
    {
        MPI_Recv(X, N, MPI_INT, (rank-1+np)%np, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	vector_sum(sum, X, N);
        //printf("%d start sending %d\n", rank, X[0]);
        MPI_Send(X, N, MPI_INT, (rank+1)%np,    101, MPI_COMM_WORLD);
        //printf("%d sent %d (ready to receive)\n", rank, X[0]);
        //printf("%d received %d\n", rank, X[0]);
        
    }
    // printf("I'm %d and my sum is ", rank);
    //print_vector(sum, N);
    
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









    /*
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
            FILE* data_file;
            data_file=fopen("ex4_data.dat","wb");

            size_t max_chunk_size = (size/np + 1);
            double* buffer = (double*) malloc(sizeof(double)*max_chunk_size);
            int i;
            size_t current_size, partial_size = size/np;

            for(i = 0; i<np; i++)
            {
                current_size = partial_size + (i<rest);

                if(i==root)
                {
                    //write root partial matrix
                    fwrite(partial_matrix, sizeof(double), current_size, data_file);
                }
                else
                {
                    MPI_Recv(buffer, current_size, MPI_DOUBLE, i, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    //printf("ricevuto il messaggio di:%d, current_size=%ld\n", i, current_size);
                    fwrite(buffer, sizeof(double), current_size, data_file);
                    //printf("scrittura compleata(%d)\n", i);
                }
                //fseek(data_file, 0, SEEK_CUR);
                //printf("testina avanti di %ld double \n", (current_size-N*N/total_np));
            }
            free(buffer);
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
    */
}
