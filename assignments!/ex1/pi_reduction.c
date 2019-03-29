#include <stdlib.h>
#include <stdio.h>
#include "omp.h"

double f(double x)
{
    return 1/(1+x*x);
}

double parallel_pi(int N, int nthreads)
{
    double h = 1./N;
    double almost_pi = 0;

    # pragma omp parallel num_threads(nthreads) reduction(+:almost_pi)
    {

        double temp = 0;

	int i;
        # pragma omp for schedule(static)
        for(i=0; i<N; i++)
            temp+=h*f(h*(i+0.5));

        almost_pi += temp;
    }
    return almost_pi*4;
}

int main( int argc, char * argv[] )
{
    const int N = (int) 1E8;
    int nthreads = argc>=2 ? atoi(argv[1]) : 4;

    double start = omp_get_wtime();
    double pi =  parallel_pi(N, nthreads);
    double end = omp_get_wtime();

    printf("pi=%.10f\ntime=%f\n", pi, end-start);

    return 0;
}
