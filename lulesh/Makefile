HOME ?= /root
CLANG_PATH ?= $(HOME)/Poseidon/llvm-project/build/bin
ENZYME_PATH ?= $(HOME)/Poseidon/Enzyme/build/Enzyme/ClangEnzyme-22.so
PROFILER_PATH ?= $(HOME)/Poseidon/Enzyme/build/Enzyme

SHELL = /bin/sh
.SUFFIXES: .cc .o

CXX = $(CLANG_PATH)/clang++
LLC = $(CLANG_PATH)/llc

SOURCES2.0 = \
	lulesh.cc \
	lulesh-viz.cc \
	lulesh-util.cc \
	lulesh-init.cc
OBJECTS2.0 = $(SOURCES2.0:.cc=.o)

CXXFLAGS = -I$(HOME)/include -L$(HOME)/lib -I/usr/include/c++/13 -I/usr/include/x86_64-linux-gnu/c++/13 -L/usr/lib/gcc/x86_64-linux-gnu/13  
CXXFLAGS += -O3 -I. -Wall -fno-exceptions -mllvm -enzyme-loose-types -fplugin=$(ENZYME_PATH) -ffast-math -march=native

all: ser-single-forward.exe ser-single-forward-prof.exe

ser-single-forward.exe: $(SOURCES2.0)
	time $(CXX) -DOMP_MERGE=0 -DUSE_MPI=0 $(SOURCES2.0) $(CXXFLAGS) -lm -o $@

clean:
	/bin/rm -f *.o *~ *.exe
	/bin/rm -rf *.dSYM
	/bin/rm -rf herbie_input_* herbie_output_*

ser-single-forward-prof.exe: $(SOURCES2.0)
	time $(CXX) -DOMP_MERGE=0 -DUSE_MPI=0 $(SOURCES2.0) $(CXXFLAGS) \
		-mllvm --fpprofile-generate \
		-L $(PROFILER_PATH) -lEnzymeFPProfile -lm -o $@

fpprofile: ser-single-forward-prof.exe
	./ser-single-forward-prof.exe -s 30 -i 100

FPOPTFLAGS = \
	-mllvm --enzyme-print \
	-mllvm --fpopt-print \
	-mllvm --fpopt-enable-herbie=1 \
	-mllvm --fpopt-enable-solver \
	-mllvm --fpopt-enable-pt=0 \
	-mllvm --fpopt-comp-cost-budget=5098952115 \
	-mllvm --fpopt-num-samples=1024 \
	-mllvm --fpopt-early-prune \
	-mllvm --herbie-num-threads=32 \
	-mllvm --fpopt-cache-path=cache \
	-mllvm --fpopt-cost-model-path=$(HOME)/Poseidon/cost-model/cm.csv \
	-mllvm --fpopt-show-table \
	-mllvm --fpprofile-use=./fpprofile

ser-single-forward-fpopt.exe: $(SOURCES2.0)
	time $(CXX) -DOMP_MERGE=0 -DUSE_MPI=0 $(SOURCES2.0) $(CXXFLAGS) $(FPOPTFLAGS) \
		-lmpfr -lm -o $@


