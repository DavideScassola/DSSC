OPT = -O3
CC = icc
MPICC = mpiicc
OMP = -qopenmp

all: pi_OMP.x pi_MPI.x
	
pi_MPI.x : pi_MPI.c
	$(MPICC) $(OPT) $^ -o $@
	
pi_OMP.x : pi_OMP.c
	$(CC) $(OPT) $(OMP) $^ -o $@

clean:
	@rm -r *.x
	
.PHONY: clean all modules        
