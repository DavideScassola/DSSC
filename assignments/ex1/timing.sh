#module load 

echo "pi_serial:"
./pi_serial.x

echo "pi_atomic:"
for i in 1 2 4 8 16 20; do
	./pi_atomic.x ${i}
done

echo "pi_critical_section:"
for i in 1 2 4 8 16 20; do
	./pi_critical_section.x $i
done

echo "pi_reduction:"
for i in 1 2 4 8 16 20; do
	./pi_reduction.x $i
done

exit
