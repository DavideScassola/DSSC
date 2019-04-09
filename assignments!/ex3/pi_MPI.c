#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

double f(double x)
{
    return 1/(1+x*x);
}

int main( int argc, char * argv[] ){

	int rank; // identifier of the process
	int total_np; // total number of MPI processes
	
	size_t N = argc<2 ? 10000000 : atoi(argv[1]); // default is 10^7
       
	double t1, t2;
		
	MPI_Init( &argc, &argv );
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	MPI_Comm_size( MPI_COMM_WORLD, &total_np );

        int reducer_proc = total_np-1; // the processor that will reduce (sum) the partial results
        int print_proc = 0 ; // the processor that will receive and print the final result 
                
	
	t1 = MPI_Wtime();
	// Local computation 
	double h = 1./N;
	int i;
	int chunk = N/total_np;
	int start = chunk*rank;
	int end = start + chunk;
	double temp;

	if(rank!=total_np-1)
		for(i=(chunk*rank);i<chunk*(rank+1);i++)
			temp+=h*f(h*(i+0.5));
	else
		for(i=(chunk*rank);i<N;i++)
	                temp+=h*f(h*(i+0.5));
	
	
	// computing the global sum
	double global_sum;
	MPI_Reduce(&temp, &global_sum, 1, MPI_DOUBLE, MPI_SUM, reducer_proc, MPI_COMM_WORLD);
	global_sum*=4;
	t2 = MPI_Wtime();
	
	// send the result to the printer processor
	if(rank==reducer_proc && print_proc!=reducer_proc)
		MPI_Send(&global_sum, 1, MPI_DOUBLE, print_proc, 101, MPI_COMM_WORLD);
	
	// receive and print the final result
	if(rank==print_proc)
	{
		if(print_proc!=reducer_proc)
			MPI_Recv(&global_sum, 1, MPI_DOUBLE, reducer_proc, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		printf("pi = %.10f, elapsed_time:%f\n", global_sum, t2-t1);
	}
	
	MPI_Finalize();

	return 0;
}
