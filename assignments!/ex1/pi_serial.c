#include <stdlib.h>
#include <stdio.h>
#include "omp.h"

double f(double x)
{
    return 1/(1+x*x);
}

double serial_pi(int N)
{
    double h = 1/N;
    double almost_pi = 0;

    for(int i=0; i<N; i++)
    {
        almost_pi+=h*f(h*(i+0.5));
    }
    
    return almost_pi*4;
}

int main( int argc, char * argv[] )
{
    const int N = (int) 1E8;
    int nthreads = argc>=2 ? atoi(argv[1]) : 4;

    double start = omp_get_wtime();
    double pi =  serial_pi(N);
    double end = omp_get_wtime();

    printf("pi=%.10f\ntime=%f\n", pi, end-start);

    return 0;
}
