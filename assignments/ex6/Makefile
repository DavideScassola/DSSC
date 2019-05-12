OPT = -O3
CC = nvcc

all: fast_transpose.x
	
fast_transpose.x : ex6.cu
	$(CC) $(OPT) $^ -o $@
	
clean:
	@rm -r *.x
	
.PHONY: clean all       
