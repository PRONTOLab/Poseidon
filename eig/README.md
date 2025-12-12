# Eigensolver Benchmark

## Running the Benchmark

```bash
python3 run_cases.py
```

This generates `biased.txt` and `equal.txt` containing the benchmark results.

## How to Read the Output

Look for entries like this in the output files:

Both `biased.txt` and `equal.txt` contain entries like:

```
Allowed error ≤ 0.1: 14.77% runtime reduction / 1.17x speedup [Budget: -96550]
  Per-eigenvalue error statistics:
    Eigenvalue 0 (100000 test cases):
      geomean: 4.300e-08 (zeros: 0/100000), median: 5.472e-08, max: 3.701e-07
      arithmean: 6.542e-08, std: 5.029e-08
    Eigenvalue 1 (100000 test cases):
      geomean: 1.806e-08 (zeros: 0/100000), median: 2.304e-08, max: 1.182e-07
      arithmean: 2.713e-08, std: 2.027e-08
    Eigenvalue 2 (100000 test cases):
      geomean: 2.020e-08 (zeros: 0/100000), median: 2.574e-08, max: 1.699e-07
      arithmean: 3.002e-08, std: 2.199e-08
  Per-eigenvector error statistics (L2 norm):
    Eigenvector 0 (100000 test cases):
      geomean: 1.421e-07 (zeros: 0/100000), median: 1.419e-07, max: 4.885e-06
      arithmean: 1.942e-07, std: 1.820e-07
    Eigenvector 1 (100000 test cases):
      geomean: 1.625e-07 (zeros: 0/100000), median: 1.617e-07, max: 3.897e-05
      arithmean: 2.298e-07, std: 3.080e-07
    Eigenvector 2 (100000 test cases):
      geomean: 1.294e-07 (zeros: 0/100000), median: 1.264e-07, max: 3.895e-05
      arithmean: 1.826e-07, std: 2.847e-07
```

### Mapping to Table I

This entry corresponds to row 3 in Table I, specifically we have:
- Speedup: `1.17x`
- $\Delta\lambda_1$: `4.3e-8 / 3.7e-7`
- $\Delta\lambda_2$: `1.8e-8 / 1.2e-7`
- $\Delta\lambda_3$: `2.0e-8 / 1.7e-7`
- $\Delta x_1$: `1.4e-7 / 4.9e-6`
- $\Delta x_2$: `1.6e-7 / 3.9e-5`
- $\Delta x_3$: `1.3e-7 / 3.9e-5`

Rows 3–4 (equal weights) can be found in `equal.txt`, and rows 5–7 (biased weights) can be found in `biased.txt`.
