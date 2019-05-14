OPT = -O4
CC = nvcc

all: fast_transpose_int.x fast_transpose_double_best.x fast_transpose_double_1024.x fast_transpose_double_512.x fast_transpose_double_64.x
	
fast_transpose_int.x : fast_transpose_int.cu
	$(CC) $(OPT) $^ -o $@

fast_transpose_double_best.x : fast_transpose_double_best.cu
	$(CC) $(OPT) $^ -o $@

fast_transpose_double_1024.x : fast_transpose_double_1024.cu
	$(CC) $(OPT) $^ -o $@

fast_transpose_double_512.x : fast_transpose_double_512.cu
	$(CC) $(OPT) $^ -o $@

fast_transpose_double_64.x : fast_transpose_double_64.cu
	$(CC) $(OPT) $^ -o $@

	
clean:
	@rm -r *.x
	
.PHONY: clean all       
