# Thinking Fast and Correct: Automated Rewriting of Numerical Code through Compiler Augmentation

---

## Overview

This repository contains the artifact for the paper "Thinking Fast and Correct: Automated Rewriting of Numerical Code through Compiler Augmentation" (CGO 2026) by Siyuan Brant Qian, Vimarsh Sathia, Ivan R. Ivanov, Jan Hückelheim, Paul Hovland, and William S. Moses.

The latest version of this artifact is available [here](https://github.com/PRONTOLab/Poseidon).

## Build From Source

### Prerequisites

```bash
sudo apt install build-essential cmake ninja-build libmpfr-dev
pip install lit numpy matplotlib tqdm
```

Additionally, install [Racket](https://racket-lang.org/) and [Rust](https://www.rust-lang.org/tools/install).

### Clone and Initialize Submodules

```bash
git clone https://github.com/PRONTOLab/Poseidon.git
cd Poseidon
git submodule update --init --recursive llvm-project Enzyme
```

### Build LLVM

```bash
cd llvm-project
mkdir build && cd build
cmake -G Ninja \
  -DLLVM_ENABLE_PROJECTS="clang" \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DCMAKE_BUILD_TYPE=Release \
  ../llvm
ninja
cd ../..
```

### Build Enzyme with Poseidon Enabled

```bash
cd Enzyme
mkdir build && cd build
cmake -G Ninja ../enzyme/ \
  -DLLVM_DIR=<...>/Poseidon/llvm-project/build/lib/cmake/llvm \
  -DLLVM_EXTERNAL_LIT=$(which lit) \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_POSEIDON=ON \
  -DCMAKE_C_COMPILER=<...>/Poseidon/llvm-project/build/bin/clang \
  -DCMAKE_CXX_COMPILER=<...>/Poseidon/llvm-project/build/bin/clang++
ninja
cd ../..
```

Replace `<...>` with the path to your Poseidon clone.

## Docker Image (Recommended)

We provide a pre-built Docker image `sbrantq/poseidon`. To start the container:

```bash
sudo docker run -it sbrantq/poseidon:latest /bin/bash
```

To copy results from the container to the host machine:
```bash
sudo docker ps -a  # Find container ID
sudo docker cp <container_id>:/root/Poseidon/<path_to_file> <host_destination>
```

**Note**: For convenience, the container ships cached outputs from external tools: [Herbie](https://github.com/herbie-fp/herbie) (`eig/cache-*` and `lulesh/cache`) and [RAPTOR](https://arxiv.org/pdf/2507.04647v1) (`dquat/dquat_gold.txt`, `eig/eig_gold.txt`). They are auxiliary and not Poseidon's results. One can remove `eig/cache-*` and `lulesh/cache` to rerun from scratch, but recomputation can take several hours.

## Reproducing Main Results

### FPBench Cost Model Ablation Study (Figure 9)

```bash
cd $HOME/Poseidon/FPBench/ablations
python3 ablation.py
```

The plot will be saved to `plots/fptaylor-extra-ex11-ablation.png`.

To copy the plot to the host machine:
```bash
sudo docker cp <container_id>:/root/Poseidon/FPBench/ablations/plots/fptaylor-extra-ex11-ablation.png .
```

### Quaternion Differentiator (Figure 10)

```bash
cd $HOME/Poseidon/dquat
python3 run_ablation.py
```

The plot will be saved to `dquat.png`.

To copy the plot to the host machine:
```bash
sudo docker cp <container_id>:/root/Poseidon/dquat/dquat.png .
```

### LULESH (Figure 13)

```bash
cd $HOME/Poseidon/lulesh
python3 ablation.py
```

The plot will be saved to `lulesh.png`.

To copy the plot to your host machine:
```bash
sudo docker cp <container_id>:/root/Poseidon/lulesh/lulesh.png .
```

**Note**: The 0-ULP result is hardware-dependent. To find the optimal configuration for your hardware, first [regenerate the cost model](#regenerating-the-cost-model), then run:

```bash
cd $HOME/Poseidon/lulesh
make && python3 run.py
python3 benchmark.py --sample-percent 10
```

This samples 10% of the optimized programs and prints a summary of all budgets achieving a ULP of less than 5 (configurable via `--ulp-threshold`). The best budget reported by the script will result in the optimal rewrites for the user's hardware and should be used in `lulesh/Makefile`.


### 3x3 Eigensolver (TABLE I)

```bash
cd $HOME/Poseidon/eig
python3 run_cases.py
```

To copy results to the host machine:
```bash
sudo docker cp <container_id>:/root/Poseidon/eig/biased.txt .
sudo docker cp <container_id>:/root/Poseidon/eig/equal.txt .
```
TABLE I entries can be found in these output files. See [`eig/README.md`](eig/README.md) for details on how to interpret the output.

## Reusing This Artifact

This section describes how to reuse this artifact to apply Poseidon to a new benchmark.

### Step 1: Set Up Paths

Configure the paths to custom builds of LLVM and Enzyme:

```bash
export CLANG_PATH=<...>/llvm-project/build/bin
export ENZYME_PATH=<...>/Enzyme/build/Enzyme/ClangEnzyme-X.so
export PROFILER_PATH=<...>/Enzyme/build/Enzyme
```

### Step 2: Profiling Pass

First, compile your program with floating-point profiling enabled to collect runtime information:

```bash
$CLANG_PATH/clang++ -O3 -ffast-math -march=native \
    -fplugin=$ENZYME_PATH \
    -mllvm --fpprofile-generate \
    -L $PROFILER_PATH -lEnzymeFPProfile \
    your_program.cc -o your_program_prof
```

### Step 3: Generate Floating-Point Profiles

Run the profiled executable with (potentially, small surrogate) inputs to generate floating-point profiles:

```bash
./your_program_prof <your_arguments>
```

This creates an `fpprofile` directory.

### Step 4: Optimization Pass

Now compile with Poseidon's optimization pass enabled:

```bash
$CLANG_PATH/clang++ -O3 -ffast-math -march=native \
    -fplugin=$ENZYME_PATH \
    -mllvm --fpprofile-use=./fpprofile \
    -mllvm --fpopt-cost-model-path=$HOME/Poseidon/cost-model/cm.csv \
    your_program.cc -o your_program_opt
```

This produces an optimized program (`your_program_opt`) that attempts to improve numerical accuracy while preserving performance.

The first run invokes external tool (e.g., Herbie) calls and performs a full dynamic-programming solve, with results cached (in the `cache` directory by default). Subsequent runs reuse these cached results to reduce execution time.

### Step 5 (Optional): Generate and Evaluate Other Optimized Programs

The first compilation generates `cache/budgets.txt` containing all achievable cost budgets from the dynamic-programming solve. To explore other performance/accuracy trade-offs:

1. **Compile with different budgets**: Recompile with varying `--fpopt-comp-cost-budget` values from `cache/budgets.txt`. Each budget produces a differently optimized binary.

2. **Benchmark**: Run each binary and compare outputs against a reference (e.g., the original program) to evaluate its performance and accuracy.

Please see `lulesh/run.py` and `lulesh/benchmark.py` for an example of automating this process.


## Miscellaneous

### Regenerating the Cost Model

The cost model (`cost-model/cm.csv`) is hardware-specific. To regenerate it for your machine:

```bash
cd $HOME/Poseidon/cost-model
python3 microbm.py
cp results.csv cm.csv
```

### Full FPBench Run

One can apply Poseidon to all FPBench programs and see statistics by

```bash
cd $HOME/Poseidon/FPBench/experiments
python3 run-all.py
python3 run.py --analytics
```

This will display maximum speedups for each error threshold and accuracy improvement statistics.

### Full LULESH Run

The following commands produce all optimized LULESH programs and perform the full performance/accuracy measurements:

```bash
cd $HOME/Poseidon/lulesh
make
python3 run.py
python3 benchmark.py
```
These can take several hours and is not required to reproduce Figure 13.

### Useful Poseidon Flags

| Flag | Description |
|------|-------------|
| `--fpprofile-use=<path>` | Path to the generated FP profile directory |
| `--fpopt-enable-herbie` | Enable Herbie for expression rewriting |
| `--fpopt-enable-pt` | Enable precision tuning
| `--fpopt-comp-cost-budget=<N>` | Cost budget for the optimization pass; overrides `enzyme_err_tol` annotation |
| `--fpopt-num-samples=<N>` | Number of samples for accuracy estimation |
| `--fpopt-early-prune` | Enable optional pruning steps in the solver |
| `--herbie-num-threads=<N>` | Number of threads for Herbie |
| `--fpopt-cache-path=<path>` | Directory to cached results (default: `cache`) |
| `--fpopt-cost-model-path=<path>` | Path to the cost model CSV |