OPT = -O3
CC = icc
MPICC = mpiicc

all: overlapping.x
	
overlapping.x : ex4_overlapping.c
	$(MPICC) $(OPT) $^ -o $@
	
clean:
	@rm -r *.x
	
.PHONY: clean all modules        
