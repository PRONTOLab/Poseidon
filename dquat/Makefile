HOME ?= /root

CLANG_PATH ?= $(HOME)/Poseidon/llvm-project/build/bin
ENZYME_PATH ?= $(HOME)/Poseidon/Enzyme/build/Enzyme/ClangEnzyme-22.so
PROFILER_PATH ?= $(HOME)/Poseidon/Enzyme/build/Enzyme

SHELL = /bin/sh

.SUFFIXES: .cpp .o

CXX = $(CLANG_PATH)/clang++

SOURCES = dquat.cpp

CXXFLAGS = -O3 -I. -Wall \
           -I$(HOME)/include \
           -L$(HOME)/lib \
           -I/usr/include/c++/13 \
           -I/usr/include/x86_64-linux-gnu/c++/13 \
           -L/usr/lib/gcc/x86_64-linux-gnu/13 \
           -fno-exceptions \
           -fpass-plugin=$(ENZYME_PATH) \
           -Xclang -load -Xclang $(ENZYME_PATH) \
           -lmpfr -ffast-math -march=native

all: dquat.exe dquat-logged.exe dquat-fpopt.exe

dquat.exe: $(SOURCES)
	@echo "Building original executable: $@"
	$(CXX) $(SOURCES) $(CXXFLAGS) -lm -o $@

dquat-logged.exe: $(SOURCES)
	@echo "Building logged executable: $@"
	$(CXX) $(SOURCES) $(CXXFLAGS) -mllvm --fpprofile-generate -L $(PROFILER_PATH) -lEnzymeFPProfile -lm -o $@

fpprofile: dquat-logged.exe
	@echo "Generating fpprofile: $<"
	./$< --profiling

dquat-fpopt.exe: dquat.cpp fpprofile
	@echo "Compiling optimized object: $@"
	$(CXX) -c $(SOURCES) $(CXXFLAGS) \
		-mllvm --fpprofile-use=./fpprofile \
		-mllvm --fpopt-print \
		-mllvm --fpopt-enable-herbie=1 \
		-mllvm --herbie-num-threads=32 \
		-mllvm --fpopt-enable-solver \
		-mllvm --fpopt-enable-pt \
		-mllvm --fpopt-comp-cost-budget=0 \
		-mllvm --fpopt-num-samples=1024 \
		-mllvm --fpopt-early-prune \
		-mllvm --enzyme-print \
		-mllvm --fpopt-show-table \
		-mllvm --fpopt-cache-path=cache \
		-mllvm --fpopt-cost-model-path=$(HOME)/Poseidon/cost-model/cm.csv \
		-mllvm --fpopt-strict-mode \
		-lm -o $@

clean:
	@echo "Cleaning up..."
	/bin/rm -f *.o *~ *.exe
	/bin/rm -rf *.dSYM
	/bin/rm -rf herbie_input_* herbie_output_*
	/bin/rm -rf fpprofile

.PHONY: all clean
