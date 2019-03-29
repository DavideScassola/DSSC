#include<stdlib.h>
#include<stdio.h>

//Header
#include<omp.h>

int main(int argc, char *argv[])
{
	//int thread_id = 0;
	int n = 100;
	int* A = (int *) malloc(sizeof(int) * n);
	int* B = (int *) malloc(sizeof(int) * n);
	int* C = (int *) malloc(sizeof(int) * n);
	
	for(int i=0;i<n;i++)
	{
		A[i]=0.0;
		B[i]=2.0;
		C[i]=0.0;
	}



#pragma omp parallel
	{
		int task_size = n/omp_get_num_threads();
		int thread_id = omp_get_thread_num();
		int start = thread_id*task_size;
		int end = start + task_size;
		printf("start=%d, end=%d",start, end);
		int i;

		for(i=start;i<end; i++)
		{
			A[i] = B[i] = thread_id;
			C[i]=A[i]+B[i];
		}
		

		//int n_threads = omp_get_num_threads();
		//thread_id = omp_get_thread_num();
		//fprintf(stdout, "I am thread %d/%d\n", thread_id, n_threads);
	}

    free(A);
    free(B);
    free(C);	


	return 0;
}
