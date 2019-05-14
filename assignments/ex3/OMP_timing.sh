cd /home/dscassol/parallel_programming/DSSC/assignments/ex3/

module load openmpi/1.8.3/intel/14.0
PROGRAM=./pi_OMP.x
echo "pi_OMP:" >> OMP_time.txt 
for i in 1 2 4 8 12 16 20 24 28 32 36 40; do
	($PROGRAM $i) >> OMP_time.txt
done

exit
