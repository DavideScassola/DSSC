cd /home/dscassol/parallel_programming/DSSC/assignments!/ex3/
module load impi-trial/5.0.1.035
PROGRAM=./pi_MPI.x
N=1000000000;
echo "PI_MPI N=${N}">>MPI_time.txt
for i in 1 2 4 8 12 16 20 24 28 32 36 40; do
	echo "nproc=${i}" >>MPI_time.txt
	(mpirun -np ${i} ${PROGRAM} ${N})>>MPI_time.txt
	done
exit
