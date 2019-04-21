#include <stdlib.h>
#include <stdio.h>
#include "omp.h"

void print_usage( int * a, int N, int nthreads ) {

  int tid, i;
  for( tid = 0; tid < nthreads; ++tid ) {

    fprintf( stdout, "%d: ", tid );

    for( i = 0; i < N; ++i ) {

      if( a[ i ] == tid) fprintf( stdout, "*" );
      else fprintf( stdout, " ");
    }
    printf("\n");
  }
}


int main(int argc, char * argv[])
{
    const int N = 120;
    int nthreads = argc>=2 ? atoi(argv[1]) : 4; // if not specified: 4 threads
    int a[N];



    ///////////// static /////////////
    printf("\nschedule: static (default chunk size)\n");
    # pragma omp parallel num_threads(nthreads)
    {
        int thread_id = omp_get_thread_num();
	int i;
        # pragma omp for schedule(static)
        for(i = 0; i < N; ++i)
            a[i] = thread_id;
    }

    print_usage(a,N,nthreads);        

    printf("\n\n");



    ///////////// static, with chunk size 1 /////////////
    printf("\nschedule: static, with chunk size 1\n");
    # pragma omp parallel num_threads(nthreads)
    {
        int thread_id = omp_get_thread_num();
	int i;
        # pragma omp for schedule(static,1)
        for(i = 0; i < N; ++i)
            a[i] = thread_id;
    }

    print_usage(a,N,nthreads);

    printf("\n\n");



    ///////////// static, with chunk size 10 /////////////
    printf("\nschedule: static, with chunk size 10\n");
    # pragma omp parallel num_threads(nthreads)
    {
        int thread_id = omp_get_thread_num();
	int i;
        # pragma omp for schedule(static,10)
        for(i = 0; i < N; ++i)
            a[i] = thread_id;
    }

    print_usage(a,N,nthreads);

    printf("\n\n");
                                                                                                                                                                                                                                       


    ///////////// dynamic /////////////
    printf("\nschedule: dynamic (default chunk size)\n");
    # pragma omp parallel num_threads(nthreads)
    {
        int thread_id = omp_get_thread_num();
	int i;
        # pragma omp for schedule(dynamic)
        for(i = 0; i < N; ++i)
            a[i] = thread_id;
    }

    print_usage(a,N,nthreads);

    printf("\n\n");



    ///////////// dynamic, with chunk size 1 /////////////
    printf("\ndynamic, with chunk size 1\n");
    # pragma omp parallel num_threads(nthreads)
    {
        int thread_id = omp_get_thread_num();
	int i;
        # pragma omp for schedule(dynamic,1)
        for(i = 0; i < N; ++i)
            a[i] = thread_id;
    }

    print_usage(a,N,nthreads);

    printf("\n\n");



    ///////////// dynamic, with chunk size 10 /////////////
    printf("\ndynamic, with chunk size 10\n");
    # pragma omp parallel num_threads(nthreads)
    {
        int thread_id = omp_get_thread_num();
	int i;
        # pragma omp for schedule(dynamic,10)
        for(i = 0; i < N; ++i)
            a[i] = thread_id;
    }

    print_usage(a,N,nthreads);

    printf("\n\n");



  return 0;
}
