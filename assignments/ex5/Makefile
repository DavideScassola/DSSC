OPT = -O3
CC = gcc
MPICC = mpicc

all: ring.x
	
ring.x : ex5.c
	$(MPICC) $(OPT) $^ -o $@
	
clean:
	@rm -r *.x
	
.PHONY: clean all       
