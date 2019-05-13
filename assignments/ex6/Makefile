OPT = -O4
CC = nvcc

all: fast_transpose_int.x fast_transpose_double.x
	
fast_transpose_int.x : fast_transpose_int.cu
	$(CC) $(OPT) $^ -o $@

fast_transpose_double.x : fast_transpose_double.cu
	$(CC) $(OPT) $^ -o $@

	
clean:
	@rm -r *.x
	
.PHONY: clean all       
