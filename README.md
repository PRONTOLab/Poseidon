# Thinking Fast and Correct: Automated Rewriting of Numerical Code through Compiler Augmentation

---

## Overview

This repository contains the artifact for the paper "Thinking Fast and Correct: Automated Rewriting of Numerical Code through Compiler Augmentation" (CGO 2026) by Siyuan Brant Qian, Vimarsh Sathia, Ivan R. Ivanov, Jan Hückelheim, Paul Hovland, and William S. Moses.

The latest version of this artifact is available [here](https://github.com/PRONTOLab/Poseidon).

## Docker Image

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