/*
 *
 * This code implements the first broadcast operation using the paradigm of message passing (MPI) 
 *
 * The broadcast operation is of kind 1:n where thre process root send a set of data to all other processes.
 * 
 * MPI_BCAST( buffer, count, datatype, root, comm )
 * [ INOUT buffer] starting address of buffer (choice)
 * [ IN count] number of entries in buffer (integer)
 * [ IN datatype] data type of buffer (handle)
 * [ IN root] rank of broadcast root (integer)
 * [ IN comm] communicator (handle)
 *
 * int MPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm )
 *
 * MPI_BCAST broadcasts a message from the process with rank root to all processes of the group, 
 * itself included. It is called by all members of group using the same arguments for comm, root. 
 * On return, the contents of root's communication buffer has been copied to all processes.
 *
 * The type signature of count, datatype on any process must be equal to the type signature of count, 
 * datatype at the root. This implies that the amount of data sent must be equal to the amount received, 
 * pairwise between each process and the root. 
 *
 * Detailed documentation of the MPI APis here:
 * https://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/node182.html
 *
 * Compile with: 
 * $mpicc bcast.c
 * 
 * Run with:
 * $mpirun -np 4 ./a.out
 *
 */

#include <stdlib.h>
#include <stdio.h>

// Header file for compiling code including MPI APIs
#include <mpi.h>

double f(double x)
{
    return 1/(1+x*x);
}

int main( int argc, char * argv[] ){

	//int rank; // store the MPI identifier of the process
	//int npes; // store the number of MPI processes

	MPI_Init( &argc, &argv );
	//MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	//MPI_Comm_size( MPI_COMM_WORLD, &npes );

//////// DO LOCAL SUM ////////////////
num_elements_per_proc = 1000;
float *rand_nums = (*float) malloc(sizeof(float)*num_elements_per_proc);
int j;
for(j=0;j<num_elements_per_proc;j++)
	rand_nums[j] = j;

// Sum the numbers locally
float local_sum = 0;
int i;
for (i = 0; i < num_elements_per_proc; i++) {
  local_sum += rand_nums[i];
}
//////// DO LOCAL SUM ////////////////


// Print  info
printf("Local sum for process %d - %f, avg = %f\n",
       world_rank, local_sum, local_sum / num_elements_per_proc);

// Reduce all of the local sums into the global sum
float global_sum;
MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0,
           MPI_COMM_WORLD);

// Print the result
if (world_rank == 0) {
  printf("Total sum = %f, avg = %f\n", global_sum,
         global_sum / (world_size * num_elements_per_proc));
}

  MPI_Finalize();
  
  return 0;

}
