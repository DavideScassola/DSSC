CXX = gcc
OPENPM = -fopenmp
OPT = -O0

all: pi_serial.x pi_atomic.x pi_critical_section.x pi_reduction.x

pi_serial.x: pi_serial.c
	$(CXX) $(OPT) -o $@ $(OPENPM) $<

pi_atomic.x: pi_atomic.c
	$(CXX) $(OPT) -o $@ $(OPENPM) $<

pi_critical_section.x: pi_critical_section.c
	$(CXX) $(OPT) -o $@ $(OPENPM) $<

pi_reduction.x: pi_reduction.c
	$(CXX) $(OPT) -o $@ $(OPENPM) $<

clean:
	@rm -r *.x


.PHONY: clean all
