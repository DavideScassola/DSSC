#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>

void pm(double* m, size_t n)
{
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
            printf("%.0f ", m[i*n+j]);
        printf("\n");
    }
}

void copy(double* out, double* in, size_t size)
{
    for(size_t i = 0; i<size; i++)
        out[i] = in[i];
}

void local2global(size_t local_i, size_t local_j, size_t* global_i, size_t* global_j, size_t rank, size_t N, int np)
{
    size_t size = N*N;
    size_t rest = size%np;
    size_t offset_to_add = rank<rest ? rank : rest;
    size_t tot = (size/np)*rank + offset_to_add  + local_i*N + local_j;
    *global_i = tot/N;
    *global_j = tot%N;
}

int main(int argc, char* argv[])
{
    /*
    double* a = (double*) malloc(sizeof(double));
    a[0]=12.45;
    FILE* f;
    f = fopen("ciao.txt","wb");
    fwrite(a, sizeof(double), 1, f);
    fclose(f);
    a[0]=0;
    f = fopen("ciao.txt","r");
    fread(a,sizeof(double),1,f);
    printf("ho letto %f\n\n", a[0]);
    fclose(f);
    */




    int rank; // identifier of the process
    int total_np; // total number of MPI processes

  	size_t N = argc<2 ? 9 : atoi(argv[1]); // default is 9
    size_t size = N*N;

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &total_np );

        //size_t local = size/total_np;
    size_t rest = size%total_np;

        //if(rank<rest)
            //local++;

        //double* partial_matrix = calloc(local, sizeof(double));
        
        /*
        size_t i = start;

        while(i<local)
        {
            partial_matrix[i]=1;
            i+=(n+1)
        }
        */

       /*
       for(int i = start; i<start+local; i++)
       {
           partial_matrix[i] = (i%n)==n ? 1 : 0;
       }
   */


   short int has_rest = rank<rest;
   size_t rows = N/total_np +1;
   size_t local_size = size/total_np+has_rest;
   double* partial_matrix = (double*) malloc(local_size * sizeof(double));

   size_t global_i, global_j;
   size_t local_i, local_j;

    //printf("pezzo n:%d\n",rank);
    for(local_i = 0; local_i<rows; local_i++)
    {
       //printf("local i = %ld, row=%ld\n", local_i,rows);
       size_t local_j_limit = (local_i<rows-1) ? N : (local_size%(N)) ;
       if(local_i==0 && local_size==N) local_j_limit = N;
       //printf("local size: %ld", local_size);
       //printf("j_lim = %ld (local i = %ld)\n", local_j_limit, local_i);

       for(local_j = 0; local_j<local_j_limit; local_j++)
       {
            local2global(local_i, local_j, &global_i, &global_j, rank, N, total_np);
            partial_matrix[local_i*N+local_j] = (global_i == global_j) ? 1 : 0;
            //printf("local(%ld,%ld, rank=%d)=global(%ld,%ld):%f\n", local_i,local_j,rank,global_i,global_j, partial_matrix[local_i*N+local_j]);
        }
        //printf("\n");
    }

    /*
    if(has_rest)
    {
        global_i = rows + rank;
        global_j = global_j_index(rank, rest, 0);
        partial_matrix[local_size-1] = (global_i == global_j) ? 1 : 0 ;
        //printf("rest=global(%ld,%ld):%f\n\n", global_i, global_j, partial_matrix[local_size-1]);
    }
    */

    if(N<10)
    {
        int root = 0;

        if(rank!=root)
        {
            MPI_Send(partial_matrix, local_size, MPI_DOUBLE, root, 101, MPI_COMM_WORLD);
            //printf("proc[%d] sent\n", rank);
        }
        else
        {
            size_t size = N*N;
            double* m = (double*) malloc(sizeof(double)*size);
            int i;
            size_t current_size, partial_size = size/total_np, starting_point = 0; 
            for(i = 0; i<total_np; i++)
            {
                current_size = partial_size + (i<rest);
                //printf("current_size=%ld, starting_point=%ld\n", current_size, starting_point);
                if(i==root)
                {
                    copy(m+starting_point, partial_matrix, current_size);
                }
                else
                {
                    //printf("aspetto messaggio da:%d\n",i);
                    MPI_Recv(m+starting_point, current_size, MPI_DOUBLE, i, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    //printf("ho scritto il messaggio di:%d, current_size=%ld, starting_point=%ld\n", i, current_size, starting_point);
                }
                starting_point+=current_size;

            }
            pm(m,N);
            free(m);    
        }
    }
    else
    {
        int root = 0;

        if(rank!=root)
        {
            MPI_Send(partial_matrix, local_size, MPI_DOUBLE, root, 101, MPI_COMM_WORLD);
        }
        else
        {
            FILE* data_file;
            data_file=fopen("ex4_data.dat","wb");

            size_t max_chunk_size = (size/total_np + 1);
            double* buffer = (double*) malloc(sizeof(double)*max_chunk_size);
            int i;
            size_t current_size, partial_size = size/total_np;

            for(i = 0; i<total_np; i++)
            {
                current_size = partial_size + (i<rest);

                if(i==root)
                {
                    //write my partial matrix
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
