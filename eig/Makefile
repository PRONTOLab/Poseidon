HOME ?= /root
CLANG_PATH ?= $(HOME)/Poseidon/llvm-project/build/bin
ENZYME_PATH ?= $(HOME)/Poseidon/Enzyme/build/Enzyme/ClangEnzyme-22.so
PROFILER_PATH ?= $(HOME)/Poseidon/Enzyme/build/Enzyme

SHELL = /bin/sh

.SUFFIXES: .cpp .o

CXX = $(CLANG_PATH)/clang++
LLC = $(CLANG_PATH)/llc

SOURCES = eig.cpp
SOURCES_F32 = eig-f32.cpp
OBJECTS = $(SOURCES:.cpp=.o)
OBJECTS_F32 = $(SOURCES_F32:.cpp=.o)

CXXFLAGS = -I$(HOME)/include -L$(HOME)/lib \
           -I/usr/include/c++/13 \
           -I/usr/include/x86_64-linux-gnu/c++/13 \
           -L/usr/lib/gcc/x86_64-linux-gnu/13 \
           -O3 -I. -Wall \
           -fno-exceptions \
           -fpass-plugin=$(ENZYME_PATH) \
           -Xclang -load -Xclang $(ENZYME_PATH) \
           -ffast-math -march=native

all: eig.exe eig-logged.exe eig-fpopt.exe eig-f32.exe

eig.exe: $(SOURCES)
	@echo "Building original executable: $@"
	time $(CXX) $(SOURCES) $(CXXFLAGS) -lm -o $@

eig-logged.exe: $(SOURCES)
	@echo "Building logged executable: $@"
	time $(CXX) $(SOURCES) $(CXXFLAGS) -mllvm --fpprofile-generate -L $(PROFILER_PATH) -lEnzymeFPProfile -lm -o $@

CASE ?= equal

fpprofile: eig-logged.exe
	./eig-logged.exe --prof --case $(CASE)

eig-fpopt.exe: eig.cpp fpprofile
	@echo "Building optimized executable: $@"
	time $(CXX) $(SOURCES) $(CXXFLAGS) \
		-mllvm --fpopt-print \
		-mllvm --fpopt-enable-herbie=1 \
		-mllvm --fpopt-enable-solver \
		-mllvm --fpopt-enable-pt \
		-mllvm --fpopt-early-prune \
		-mllvm --fpopt-reduction-eval=geomean \
		-mllvm --fpopt-num-samples=1024 \
		-mllvm --fpopt-show-table \
		-mllvm --fpopt-cache-path=cache \
		-mllvm --fpopt-aggressive-dce \
		-mllvm --fpopt-max-expr-length=9999999 \
		-mllvm --fpopt-comp-cost-budget=0 \
		-mllvm --fpopt-cost-model-path=$(HOME)/Poseidon/cost-model/cm.csv \
		-mllvm --fpprofile-use=./fpprofile \
		-lm -o $@

eig-f32.exe: $(SOURCES_F32)
	@echo "Building f32 (float) executable: $@"
	time $(CXX) $(SOURCES_F32) $(CXXFLAGS) -lm -o $@

clean:
	@echo "Cleaning up..."
	/bin/rm -f *.o *~ *.exe
	/bin/rm -rf *.dSYM
	/bin/rm -rf herbie_input_* herbie_output_*
	/bin/rm -rf fpprofile

.PHONY: all clean
