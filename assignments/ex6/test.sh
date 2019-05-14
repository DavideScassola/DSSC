cd /home/dscassol/parallel_programming/DSSC/assignments/ex6

module load cudatoolkit

printf "\n\n\n" >> results.txt
echo "./fast_transpose_int.x" >> results.txt
./fast_transpose_int.x >> results.txt

printf "\n\n\n" >> results.txt
echo "./fast_transpose_double_64.x" >> results.txt
./fast_transpose_double_64.x >> results.txt

printf "\n\n\n" >> results.txt
echo "./fast_transpose_double_512.x" >> results.txt
./fast_transpose_double_512.x >> results.txt

printf "\n\n\n" >> results.txt
echo "./fast_transpose_double_1024.x">> results.txt
./fast_transpose_double_1024.x >> results.txt

printf "\n\n\n" >> results.txt
echo "./fast_transpose_double_best.x">> results.txt
./fast_transpose_double_best.x >> results.txt

printf "\n\n\n" >> results.txt
echo "./naive_transpose_double.x">> results.txt
./naive_transpose_double.x >> results.txt

rm test.sh.*
exit
