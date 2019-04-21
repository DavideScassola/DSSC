CXX = gcc
OPENPM = -fopenmp

all: see_threads.x

see_threads.x: ex2.c
	$(CXX) -o $@ $(OPENPM) $<

clean:
	@rm -r *.x


.PHONY: clean all
