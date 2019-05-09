OPT = -O3
MPICC = mpiicc

all: ring_overlapping.x ring_sequential.x
	
ring_overlapping.x : ex5_overlapping.c
	$(MPICC) $(OPT) $^ -o $@

ring_sequential.x : ex5_sequential.c
	$(MPICC) $(OPT) $^ -o $@
	
clean:
	@rm -r *.x
	
.PHONY: clean all       
