#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
import pickle
import numpy as np
import re
import multiprocessing
from tqdm import tqdm
import math

TIMEOUT_SECONDS = 1000
NUM_RUNS = 10


def geomean(values):
    assert len(values) > 0, "Cannot compute geometric mean of an empty list"
    sum_log = 0.0
    nonzero_count = 0
    zero_count = 0

    for x in values:
        if x != 0:
            sum_log += math.log(x)
            nonzero_count += 1
        else:
            zero_count += 1

    if nonzero_count == 0:
        return (0.0, len(values))

    return (math.exp(sum_log / nonzero_count), zero_count)


def run_command(
    command, description, capture_output=False, output_file=None, verbose=True, env=None, timeout=TIMEOUT_SECONDS
):
    try:
        if capture_output and output_file:
            with open(output_file, "w") as f:
                subprocess.run(
                    command,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=True,
                    env=env,
                    timeout=timeout,
                )
            return None
        elif capture_output:
            result = subprocess.run(command, capture_output=True, text=True, check=True, env=env, timeout=timeout)
            return result.stdout
        else:
            if verbose:
                subprocess.check_call(command, env=env, timeout=timeout)
            else:
                subprocess.check_call(
                    command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, timeout=timeout
                )
    except subprocess.TimeoutExpired:
        print(f"Timeout after {timeout} seconds during: {description}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error during: {description}")
        if capture_output and output_file:
            print(f"Check the output file: {output_file} for details.")
        else:
            print(e)
        return None


def measure_runtime(executable, num_runs):
    print(f"=== Measuring runtime for {executable} ===")
    runtimes = []
    for i in range(num_runs):
        cmd = [executable]
        result = run_command(cmd, f"Running {executable} (run {i+1}/{num_runs})", capture_output=True, verbose=False)
        if result is None:
            runtimes.append(np.nan)
            continue
        runtime_line = next((line for line in result.split("\n") if "Elapsed time" in line), None)
        if runtime_line:
            match = re.search(r"Elapsed time\s*=\s*(\S+)", runtime_line)
            if match:
                try:
                    runtime = float(match.group(1))
                    print(f"Extracted runtime: {runtime}, iter {i+1}/{num_runs}")
                    runtimes.append(runtime)
                except ValueError:
                    print(f"Invalid runtime value extracted from output of {executable}")
                    runtimes.append(np.nan)
            else:
                print(f"Could not parse runtime from output of {executable}")
                runtimes.append(np.nan)
        else:
            print(f"Could not find 'Elapsed time' line in output of {executable}")
            runtimes.append(np.nan)
    if not any(~np.isnan(runtimes)):
        print(f"No valid runtimes collected for {executable}.")
        return np.nan
    average_runtime = np.nanmean(runtimes)
    print(f"Average runtime for {executable}: {average_runtime:.6f} seconds")
    return average_runtime


def collect_output(executable, output_file):
    cmd = [executable, "--output-path", output_file]
    run_command(
        cmd, f"Running {executable} and collecting output", capture_output=False, output_file=None, verbose=False
    )


def compute_error_geometric_average(
    reference_output,
    test_output,
    return_max=False,
    return_stats=False,
    return_per_eigenvalue=False,
    compute_eigenvectors=False,
):
    ref_eigenvalues_dict = extract_all_eigenvalues(reference_output)
    test_eigenvalues_dict = extract_all_eigenvalues(test_output)

    if compute_eigenvectors:
        ref_eigenvectors_dict = extract_all_eigenvectors(reference_output)
        test_eigenvectors_dict = extract_all_eigenvectors(test_output)

    if ref_eigenvalues_dict is None or test_eigenvalues_dict is None:
        print("Could not extract eigenvalues for error computation.")
        if compute_eigenvectors:
            return None, None
        if return_per_eigenvalue:
            return None
        if return_stats:
            return None, None, None
        return None if not return_max else (None, None)

    if set(ref_eigenvalues_dict.keys()) != set(test_eigenvalues_dict.keys()):
        print("Mismatch in iteration numbers between reference and test outputs.")
        if compute_eigenvectors:
            return None, None
        if return_per_eigenvalue:
            return None
        if return_stats:
            return None, None, None
        return None if not return_max else (None, None)

    if compute_eigenvectors:
        if ref_eigenvectors_dict is None or test_eigenvectors_dict is None:
            print("Could not extract eigenvectors for error computation.")
            return None, None
        if set(ref_eigenvectors_dict.keys()) != set(test_eigenvectors_dict.keys()):
            print("Mismatch in iteration numbers between reference and test eigenvector outputs.")
            return None, None

    all_relative_diffs = []
    per_eigenvalue_diffs = {}

    if compute_eigenvectors:
        per_eigenvector_diffs = {}

    for iteration in ref_eigenvalues_dict:
        ref_eigenvalues = ref_eigenvalues_dict[iteration]
        test_eigenvalues = test_eigenvalues_dict.get(iteration, [])

        if not ref_eigenvalues or not test_eigenvalues:
            print(f"Missing eigenvalues for iteration {iteration}. Skipping.")
            continue

        if len(ref_eigenvalues) != len(test_eigenvalues):
            print(f"Mismatch in number of eigenvalues at iteration {iteration}. Skipping.")
            continue

        for idx, (ref_val, test_val) in enumerate(zip(ref_eigenvalues, test_eigenvalues)):
            if np.isnan(ref_val) or np.isnan(test_val):
                continue
            if abs(ref_val) < 1e-15:
                continue
            relative_diff = abs(ref_val - test_val) / abs(ref_val)
            all_relative_diffs.append(relative_diff)

            if idx not in per_eigenvalue_diffs:
                per_eigenvalue_diffs[idx] = []
            per_eigenvalue_diffs[idx].append(relative_diff)

        if compute_eigenvectors:
            ref_eigenvectors = ref_eigenvectors_dict.get(iteration, [])
            test_eigenvectors = test_eigenvectors_dict.get(iteration, [])

            if ref_eigenvectors and test_eigenvectors and len(ref_eigenvectors) == len(test_eigenvectors):
                for vec_idx, (ref_vec, test_vec) in enumerate(zip(ref_eigenvectors, test_eigenvectors)):
                    if len(ref_vec) == len(test_vec) and len(ref_vec) == 3:
                        ref_norm = np.sqrt(sum(x**2 for x in ref_vec if not np.isnan(x)))
                        diff_norm = np.sqrt(
                            sum((r - t) ** 2 for r, t in zip(ref_vec, test_vec) if not np.isnan(r) and not np.isnan(t))
                        )

                        if ref_norm > 1e-15:
                            relative_diff = diff_norm / ref_norm
                            if vec_idx not in per_eigenvector_diffs:
                                per_eigenvector_diffs[vec_idx] = []
                            per_eigenvector_diffs[vec_idx].append(relative_diff)

    if compute_eigenvectors:
        eigenvector_stats = {}
        for idx in sorted(per_eigenvector_diffs.keys()):
            diffs = per_eigenvector_diffs[idx]
            if diffs:
                try:
                    geomean_val, zero_count = geomean(diffs)
                    eigenvector_stats[idx] = {
                        "median": np.median(diffs),
                        "geomean": geomean_val,
                        "zero_count": zero_count,
                        "arithmean": np.mean(diffs),
                        "std": np.std(diffs),
                        "min": np.min(diffs),
                        "max": np.max(diffs),
                        "count": len(diffs),
                    }
                except (ValueError, OverflowError) as e:
                    print(f"Error computing stats for eigenvector {idx}: {e}")
                    eigenvector_stats[idx] = None

        eigenvalue_stats = {}
        for idx in sorted(per_eigenvalue_diffs.keys()):
            diffs = per_eigenvalue_diffs[idx]
            if diffs:
                try:
                    geomean_val, zero_count = geomean(diffs)
                    eigenvalue_stats[idx] = {
                        "median": np.median(diffs),
                        "geomean": geomean_val,
                        "zero_count": zero_count,
                        "arithmean": np.mean(diffs),
                        "std": np.std(diffs),
                        "min": np.min(diffs),
                        "max": np.max(diffs),
                        "count": len(diffs),
                    }
                except (ValueError, OverflowError) as e:
                    print(f"Error computing stats for eigenvalue {idx}: {e}")
                    eigenvalue_stats[idx] = None

        return eigenvalue_stats, eigenvector_stats

    if return_per_eigenvalue:
        per_eigenvalue_stats = {}
        for idx in sorted(per_eigenvalue_diffs.keys()):
            diffs = per_eigenvalue_diffs[idx]
            if diffs:
                try:
                    geomean_val, zero_count = geomean(diffs)
                    max_err = np.max(diffs)
                    percentile_under_20pct = None
                    if max_err > 1.0:
                        errors_under_20pct = sum(1 for e in diffs if e < 0.2)
                        percentile_under_20pct = (errors_under_20pct / len(diffs)) * 100

                    per_eigenvalue_stats[idx] = {
                        "median": np.median(diffs),
                        "geomean": geomean_val,
                        "zero_count": zero_count,
                        "arithmean": np.mean(diffs),
                        "std": np.std(diffs),
                        "min": np.min(diffs),
                        "max": max_err,
                        "count": len(diffs),
                        "percentile_under_20pct": percentile_under_20pct,
                    }
                except (ValueError, OverflowError) as e:
                    print(f"Error computing stats for eigenvalue {idx}: {e}")
                    per_eigenvalue_stats[idx] = None
        return per_eigenvalue_stats

    if not all_relative_diffs:
        print("No valid relative differences to compute error.")
        if return_stats:
            return np.nan, np.nan, None
        return np.nan if not return_max else (np.nan, np.nan)

    try:
        geomean_error, zero_count = geomean(all_relative_diffs)
        arithmean_error = np.mean(all_relative_diffs)
        if return_stats:
            max_err = np.max(all_relative_diffs)
            percentile_under_20pct = None
            if max_err > 1.0:
                errors_under_20pct = sum(1 for e in all_relative_diffs if e < 0.2)
                percentile_under_20pct = (errors_under_20pct / len(all_relative_diffs)) * 100

            error_stats = {
                "median": np.median(all_relative_diffs),
                "geomean": geomean_error,
                "zero_count": zero_count,
                "arithmean": arithmean_error,
                "std": np.std(all_relative_diffs),
                "min": np.min(all_relative_diffs),
                "max": max_err,
                "count": len(all_relative_diffs),
                "percentile_under_20pct": percentile_under_20pct,
            }
            return geomean_error, max(all_relative_diffs), error_stats
        if return_max:
            max_error = max(all_relative_diffs)
            return geomean_error, max_error
        return geomean_error
    except (ValueError, OverflowError) as e:
        print(f"Error computing geometric average: {e}")
        if return_stats:
            return np.nan, np.nan, None
        return np.nan if not return_max else (np.nan, np.nan)


def extract_all_eigenvalues(output_file):
    eigenvalues_dict = {}
    current_iteration = None

    with open(output_file, "r") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        iter_match = re.match(r"### Iteration:\s*(\d+)", line)
        if iter_match:
            current_iteration = int(iter_match.group(1))
            eigenvalues_dict[current_iteration] = []
            continue

        if current_iteration is not None:
            if line.strip() == "Eigenvalues:" and idx + 1 < len(lines):
                evals_line = lines[idx + 1].strip()
                m = re.search(r"\[(.*)\]", evals_line)
                if m:
                    evals_str = m.group(1)
                    evals = []
                    for val_str in evals_str.split(","):
                        val_str = val_str.strip()
                        if val_str.lower() in ["nan", "-nan"]:
                            evals.append(np.nan)
                        else:
                            try:
                                evals.append(float(val_str))
                            except ValueError:
                                print(
                                    f"Invalid eigenvalue '{val_str}' in {output_file} at iteration {current_iteration}"
                                )
                                evals.append(np.nan)
                    eigenvalues_dict[current_iteration] = evals
                else:
                    print(f"Regex did not match for eigenvalue line in {output_file} at iteration {current_iteration}")
    if not eigenvalues_dict:
        print(f"No eigenvalues found in {output_file}")
        return None
    return eigenvalues_dict


def extract_all_eigenvectors(output_file):
    eigenvectors_dict = {}
    current_iteration = None

    with open(output_file, "r") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        iter_match = re.match(r"### Iteration:\s*(\d+)", line)
        if iter_match:
            current_iteration = int(iter_match.group(1))
            eigenvectors_dict[current_iteration] = []
            continue

        if current_iteration is not None:
            if line.strip() == "Eigenvectors:" and idx + 1 < len(lines):
                evecs = []
                for vec_idx in range(3):
                    if idx + vec_idx + 1 < len(lines):
                        evec_line = lines[idx + vec_idx + 1].strip()
                        m = re.search(r"\[(.*)\]", evec_line)
                        if m:
                            evec_str = m.group(1)
                            evec = []
                            for val_str in evec_str.split(","):
                                val_str = val_str.strip()
                                if val_str.lower() in ["nan", "-nan"]:
                                    evec.append(np.nan)
                                else:
                                    try:
                                        evec.append(float(val_str))
                                    except ValueError:
                                        print(
                                            f"Invalid eigenvector component '{val_str}' in {output_file} at iteration {current_iteration}"
                                        )
                                        evec.append(np.nan)
                            evecs.append(evec)
                        else:
                            print(
                                f"Regex did not match for eigenvector line in {output_file} at iteration {current_iteration}"
                            )
                            evecs.append([np.nan, np.nan, np.nan])
                eigenvectors_dict[current_iteration] = evecs

    if not eigenvectors_dict:
        print(f"No eigenvectors found in {output_file}")
        return None
    return eigenvectors_dict


def accuracy_task(executable, reference_output="eig_gold.txt"):
    m = re.match(r".*eig-fpopt-(.*)\.exe", executable)
    if m:
        budget = int(m.group(1))
    else:
        print(f"Could not extract budget from executable name: {executable}")
        return None, None, None, None, None, None, None
    output_file = os.path.join("outputs", f"output_budget_{budget}.txt")
    collect_output(executable, output_file)

    error_result = compute_error_geometric_average(reference_output, output_file, return_max=True, return_stats=True)
    per_eigenvalue_stats = compute_error_geometric_average(reference_output, output_file, return_per_eigenvalue=True)
    eigenvector_result = compute_error_geometric_average(reference_output, output_file, compute_eigenvectors=True)
    if eigenvector_result and eigenvector_result != (None, None):
        per_eigenvalue_stats_from_evec, per_eigenvector_stats = eigenvector_result
    else:
        per_eigenvector_stats = None

    if error_result is not None and error_result != (None, None, None):
        error, max_error, error_stats = error_result
        if error is not None:
            print(f"Relative error for budget {budget}: geomean={error:.3e} (worst case: {max_error:.3e})")

            if per_eigenvalue_stats:
                print(f"  Per-eigenvalue errors for budget {budget}:")
                for idx in sorted(per_eigenvalue_stats.keys()):
                    if per_eigenvalue_stats[idx]:
                        stats = per_eigenvalue_stats[idx]
                        print(
                            f"    Eigenvalue {idx}: geomean={stats['geomean']:.3e} (zeros: {stats.get('zero_count', 0)}/{stats['count']}), max={stats['max']:.3e}"
                        )

            if per_eigenvector_stats:
                print(f"  Per-eigenvector errors (L2 norm) for budget {budget}:")
                for idx in sorted(per_eigenvector_stats.keys()):
                    if per_eigenvector_stats[idx]:
                        stats = per_eigenvector_stats[idx]
                        print(
                            f"    Eigenvector {idx}: geomean={stats['geomean']:.3e} (zeros: {stats.get('zero_count', 0)}/{stats['count']}), max={stats['max']:.3e}"
                        )
        else:
            print(f"Relative error for budget {budget} could not be computed.")
            error = None
            max_error = None
            error_stats = None
            per_eigenvalue_stats = None
            per_eigenvector_stats = None
    else:
        print(f"Relative error for budget {budget} could not be computed.")
        error = None
        max_error = None
        error_stats = None
        per_eigenvalue_stats = None
        per_eigenvector_stats = None

    os.remove(output_file)
    return budget, error, max_error, error_stats, per_eigenvalue_stats, per_eigenvector_stats


def runtime_task(executable):
    m = re.match(r".*eig-fpopt-(.*)\.exe", executable)
    if m:
        budget = int(m.group(1))
    else:
        print(f"Could not extract budget from executable name: {executable}")
        return None, None
    runtime = measure_runtime(executable, num_runs=NUM_RUNS)
    if runtime is not None:
        print(f"Measured runtime for budget {budget}: {runtime}")
    else:
        print(f"Runtime for budget {budget} could not be measured.")
    return budget, runtime


def threshold_formula(per_eigenvalue_stats):
    eval_err = {}
    if per_eigenvalue_stats:
        for idx in per_eigenvalue_stats:
            if per_eigenvalue_stats[idx] and "geomean" in per_eigenvalue_stats[idx]:
                eval_err[idx] = per_eigenvalue_stats[idx]["geomean"]

    result = eval_err.get(0, 0)
    return result if result > 0 else None


def analyze_data(data, thresholds=None):
    budgets = data["budgets"]
    runtimes = data["runtimes"]
    errors = data["errors"]
    max_errors = data.get("max_errors", [None] * len(errors))
    error_stats_per_budget = data.get("error_stats", {})
    per_eigenvalue_stats_per_budget = data.get("per_eigenvalue_stats", {})
    per_eigenvector_stats_per_budget = data.get("per_eigenvector_stats", {})
    original_runtime = data["original_runtime"]
    original_error = data["original_error"]
    original_error_stats = data.get("original_error_stats", None)
    original_per_eigenvalue_stats = data.get("original_per_eigenvalue_stats", None)
    original_per_eigenvector_stats = data.get("original_per_eigenvector_stats", None)

    f32_runtime = data.get("f32_runtime", None)
    f32_error = data.get("f32_error", None)
    f32_max_error = data.get("f32_max_error", None)
    f32_error_stats = data.get("f32_error_stats", None)
    f32_per_eigenvalue_stats = data.get("f32_per_eigenvalue_stats", None)
    f32_per_eigenvector_stats = data.get("f32_per_eigenvector_stats", None)

    print(f"Original Program (double precision): {original_runtime:.6f} seconds, {original_error:.6e}")
    if original_error_stats:
        print(f"  Original error distribution ({original_error_stats['count']} test cases):")
        print(
            f"    median: {original_error_stats['median']:.3e}, geomean: {original_error_stats['geomean']:.3e} (zeros: {original_error_stats.get('zero_count', 0)}/{original_error_stats['count']}), arithmean: {original_error_stats['arithmean']:.3e}"
        )
        print(f"    std: {original_error_stats['std']:.3e}, max: {original_error_stats['max']:.3e}")

    if original_per_eigenvalue_stats:
        print(f"\n  Original per-eigenvalue errors:")
        for idx in sorted(original_per_eigenvalue_stats.keys()):
            if original_per_eigenvalue_stats[idx]:
                stats = original_per_eigenvalue_stats[idx]
                print(
                    f"    Eigenvalue {idx}: geomean={stats['geomean']:.3e} (zeros: {stats.get('zero_count', 0)}/{stats['count']}), max={stats['max']:.3e}"
                )

    if original_per_eigenvector_stats:
        print(f"\n  Original per-eigenvector errors (L2 norm):")
        for idx in sorted(original_per_eigenvector_stats.keys()):
            if original_per_eigenvector_stats[idx]:
                stats = original_per_eigenvector_stats[idx]
                print(
                    f"    Eigenvector {idx}: geomean={stats['geomean']:.3e} (zeros: {stats.get('zero_count', 0)}/{stats['count']}), max={stats['max']:.3e}"
                )

    if f32_runtime is not None and not np.isnan(f32_runtime):
        print(f"\nf32 Program (single precision): {f32_runtime:.6f} seconds, {f32_error:.6e}")
        if f32_error_stats:
            print(f"  f32 error distribution ({f32_error_stats['count']} test cases):")
            print(
                f"    median: {f32_error_stats['median']:.3e}, geomean: {f32_error_stats['geomean']:.3e} (zeros: {f32_error_stats.get('zero_count', 0)}/{f32_error_stats['count']}), arithmean: {f32_error_stats['arithmean']:.3e}"
            )
            print(f"    std: {f32_error_stats['std']:.3e}, max: {f32_error_stats['max']:.3e}")

        if f32_per_eigenvalue_stats:
            print(f"\n  f32 per-eigenvalue errors:")
            for idx in sorted(f32_per_eigenvalue_stats.keys()):
                if f32_per_eigenvalue_stats[idx]:
                    stats = f32_per_eigenvalue_stats[idx]
                    print(
                        f"    Eigenvalue {idx}: geomean={stats['geomean']:.3e} (zeros: {stats.get('zero_count', 0)}/{stats['count']}), max={stats['max']:.3e}"
                    )

        if f32_per_eigenvector_stats:
            print(f"\n  f32 per-eigenvector errors (L2 norm):")
            for idx in sorted(f32_per_eigenvector_stats.keys()):
                if f32_per_eigenvector_stats[idx]:
                    stats = f32_per_eigenvector_stats[idx]
                    print(
                        f"    Eigenvector {idx}: geomean={stats['geomean']:.3e} (zeros: {stats.get('zero_count', 0)}/{stats['count']}), max={stats['max']:.3e}"
                    )

        if original_runtime is not None:
            speedup = original_runtime / f32_runtime
            print(f"\n  f32 vs double comparison:")
            print(f"    Runtime: {speedup:.2f}x speedup")
            if original_error is not None and f32_error is not None:
                error_increase = f32_error / original_error
                print(f"    Error: {error_increase:.2f}x higher (worse accuracy)")

    if thresholds is None:
        thresholds = [0, 1e-14, 1e-12, 1e-10, 1e-9, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 0.2, 0.3, 0.4, 0.5, 0.9, 1]

    threshold_errors = []
    for budget in budgets:
        per_ev_stats = per_eigenvalue_stats_per_budget.get(budget, None)
        threshold_value = threshold_formula(per_ev_stats)
        threshold_errors.append(threshold_value if threshold_value is not None else np.nan)

    min_runtime_ratios = {}
    worst_case_errors = {}
    error_statistics = {}
    per_eigenvalue_statistics = {}
    per_eigenvector_statistics = {}
    for threshold in thresholds:
        min_ratio = None
        best_program_idx = None
        best_program_budget = None
        for idx, (threshold_err, runtime, budget) in enumerate(zip(threshold_errors, runtimes, budgets)):
            if (
                threshold_err is not None
                and not np.isnan(threshold_err)
                and runtime is not None
                and threshold_err <= threshold
            ):
                runtime_ratio = runtime / original_runtime
                if min_ratio is None or runtime_ratio < min_ratio:
                    min_ratio = runtime_ratio
                    best_program_idx = idx
                    best_program_budget = budget
        if min_ratio is not None:
            min_runtime_ratios[threshold] = min_ratio
            if best_program_idx is not None and best_program_idx < len(max_errors):
                worst_case_errors[threshold] = max_errors[best_program_idx]
            else:
                worst_case_errors[threshold] = None

            if best_program_budget in error_stats_per_budget:
                error_statistics[threshold] = error_stats_per_budget[best_program_budget]

            if best_program_budget in per_eigenvalue_stats_per_budget:
                per_eigenvalue_statistics[threshold] = per_eigenvalue_stats_per_budget[best_program_budget]

            if best_program_budget in per_eigenvector_stats_per_budget:
                per_eigenvector_statistics[threshold] = per_eigenvector_stats_per_budget[best_program_budget]

    overall_runtime_improvements = {}
    for threshold in thresholds:
        ratio = min_runtime_ratios.get(threshold)
        if ratio is not None:
            percentage_improvement = (1 - ratio) * 100
            overall_runtime_improvements[threshold] = percentage_improvement
        else:
            overall_runtime_improvements[threshold] = None

    valid_errors = [e for e in errors if e is not None and not np.isnan(e)]
    if valid_errors and original_error is not None and not np.isnan(original_error):
        min_error = min(valid_errors)
        min_error_idx = errors.index(min_error)
        best_program_max_error = (
            max_errors[min_error_idx]
            if min_error_idx < len(max_errors) and max_errors[min_error_idx] is not None
            else None
        )
        best_accuracy_runtime = (
            runtimes[min_error_idx] if min_error_idx < len(runtimes) and runtimes[min_error_idx] is not None else None
        )
        max_accuracy_improvement = (original_error - min_error) / original_error * 100

        if best_program_max_error is not None:
            print(
                f"\nMaximum accuracy improvement: {max_accuracy_improvement:.2f}% (original error: {original_error:.6e}, best error: {min_error:.6e}, worst case: {best_program_max_error:.6e})"
            )
        else:
            print(
                f"\nMaximum accuracy improvement: {max_accuracy_improvement:.2f}% (original error: {original_error:.6e}, best error: {min_error:.6e})"
            )

        if best_accuracy_runtime is not None and original_runtime is not None:
            slowdown_percentage = ((best_accuracy_runtime - original_runtime) / original_runtime) * 100
            speedup_factor = best_accuracy_runtime / original_runtime
            if slowdown_percentage > 0:
                print(f"  Cost: {slowdown_percentage:.2f}% slowdown ({speedup_factor:.2f}x slower)")
            else:
                print(f"  Cost: {abs(slowdown_percentage):.2f}% speedup ({1/speedup_factor:.2f}x faster)")

    most_accurate_idx = None
    min_error = float("inf")
    for idx, err in enumerate(errors):
        if err is not None and not np.isnan(err) and err < min_error:
            min_error = err
            most_accurate_idx = idx

    if most_accurate_idx is not None:
        most_accurate_budget = budgets[most_accurate_idx]
        most_accurate_runtime = runtimes[most_accurate_idx]
        most_accurate_stats = error_stats_per_budget.get(most_accurate_budget, None)
        most_accurate_per_eigenvalue = per_eigenvalue_stats_per_budget.get(most_accurate_budget, None)
        most_accurate_per_eigenvector = per_eigenvector_stats_per_budget.get(most_accurate_budget, None)

        print(f"\nMost Accurate Program (Budget {most_accurate_budget}):")
        print(
            f"  Runtime: {most_accurate_runtime:.6f} seconds ({most_accurate_runtime/original_runtime:.2f}x of original)"
        )
        print(f"  Geomean error: {min_error:.6e}")
        if most_accurate_stats:
            print(f"  Error distribution ({most_accurate_stats['count']} test cases):")
            print(
                f"    median: {most_accurate_stats['median']:.3e}, geomean: {most_accurate_stats['geomean']:.3e} (zeros: {most_accurate_stats.get('zero_count', 0)}/{most_accurate_stats['count']}), arithmean: {most_accurate_stats['arithmean']:.3e}"
            )
            print(f"    std: {most_accurate_stats['std']:.3e}, max: {most_accurate_stats['max']:.3e}")

        if most_accurate_per_eigenvalue:
            print(f"\n  Per-eigenvalue errors for most accurate program:")
            for idx in sorted(most_accurate_per_eigenvalue.keys()):
                if most_accurate_per_eigenvalue[idx]:
                    stats = most_accurate_per_eigenvalue[idx]
                    print(
                        f"    Eigenvalue {idx}: geomean={stats['geomean']:.3e} (zeros: {stats.get('zero_count', 0)}/{stats['count']}), max={stats['max']:.3e}"
                    )

        if most_accurate_per_eigenvector:
            print(f"\n  Per-eigenvector errors (L2 norm) for most accurate program:")
            for idx in sorted(most_accurate_per_eigenvector.keys()):
                if most_accurate_per_eigenvector[idx]:
                    stats = most_accurate_per_eigenvector[idx]
                    print(
                        f"    Eigenvector {idx}: geomean={stats['geomean']:.3e} (zeros: {stats.get('zero_count', 0)}/{stats['count']}), max={stats['max']:.3e}"
                    )

            if original_per_eigenvalue_stats:
                print(f"\n  Per-eigenvalue comparison with original (percentage change, negative = improvement):")
                for idx in sorted(most_accurate_per_eigenvalue.keys()):
                    if (
                        idx in original_per_eigenvalue_stats
                        and most_accurate_per_eigenvalue[idx]
                        and original_per_eigenvalue_stats[idx]
                    ):
                        orig_stats = original_per_eigenvalue_stats[idx]
                        new_stats = most_accurate_per_eigenvalue[idx]
                        geomean_change = (new_stats["geomean"] - orig_stats["geomean"]) / orig_stats["geomean"] * 100
                        max_change = (new_stats["max"] - orig_stats["max"]) / orig_stats["max"] * 100
                        print(f"    Eigenvalue {idx}: geomean {geomean_change:+.1f}%, max {max_change:+.1f}%")

            if original_per_eigenvector_stats and most_accurate_per_eigenvector:
                print(f"\n  Per-eigenvector comparison with original (percentage change, negative = improvement):")
                for idx in sorted(most_accurate_per_eigenvector.keys()):
                    if (
                        idx in original_per_eigenvector_stats
                        and most_accurate_per_eigenvector[idx]
                        and original_per_eigenvector_stats[idx]
                    ):
                        orig_stats = original_per_eigenvector_stats[idx]
                        new_stats = most_accurate_per_eigenvector[idx]
                        geomean_change = (new_stats["geomean"] - orig_stats["geomean"]) / orig_stats["geomean"] * 100
                        max_change = (new_stats["max"] - orig_stats["max"]) / orig_stats["max"] * 100
                        print(f"    Eigenvector {idx}: geomean {geomean_change:+.1f}%, max {max_change:+.1f}%")

            if original_error_stats:
                print(f"\n  Comparison with Original (percentage change, negative = improvement):")

                if (
                    most_accurate_stats["median"] == 0
                    or np.isnan(most_accurate_stats["median"])
                    or original_error_stats["median"] == 0
                ):
                    print(
                        f"    DEBUG: most_accurate median={most_accurate_stats['median']}, original median={original_error_stats['median']}"
                    )

                median_change = (
                    (
                        (most_accurate_stats["median"] - original_error_stats["median"])
                        / original_error_stats["median"]
                        * 100
                    )
                    if original_error_stats["median"] != 0 and not np.isnan(most_accurate_stats["median"])
                    else float("nan")
                )
                geomean_change = (
                    (most_accurate_stats["geomean"] - original_error_stats["geomean"])
                    / original_error_stats["geomean"]
                    * 100
                )
                arithmean_change = (
                    (most_accurate_stats["arithmean"] - original_error_stats["arithmean"])
                    / original_error_stats["arithmean"]
                    * 100
                )
                max_change = (
                    (most_accurate_stats["max"] - original_error_stats["max"]) / original_error_stats["max"] * 100
                )

                print(
                    f"    Median: {median_change:+.1f}%"
                    if not np.isnan(median_change)
                    else "    Median: N/A (check DEBUG output)"
                )
                print(f"    Geomean: {geomean_change:+.1f}%")
                print(f"    Arithmean: {arithmean_change:+.1f}%")
                print(f"    Max error: {max_change:+.1f}%")

    print("\nPercentage of runtime improvements while allowing some level of relative error:")
    print("(Budget values are encoded in executable filenames as eig-fpopt-<budget>.exe)")
    for threshold in thresholds:
        percentage_improvement = overall_runtime_improvements[threshold]
        if percentage_improvement is not None:
            best_budget = None
            for idx, (threshold_err, runtime, budget) in enumerate(zip(threshold_errors, runtimes, budgets)):
                if (
                    threshold_err is not None
                    and not np.isnan(threshold_err)
                    and runtime is not None
                    and threshold_err <= threshold
                ):
                    runtime_ratio = runtime / original_runtime
                    if min_runtime_ratios[threshold] == runtime_ratio:
                        best_budget = budget
                        break

            worst_error = worst_case_errors.get(threshold)
            stats = error_statistics.get(threshold)
            per_ev_stats = per_eigenvalue_statistics.get(threshold)
            per_evec_stats = per_eigenvector_statistics.get(threshold)

            speedup_str = f"{percentage_improvement:.2f}% runtime reduction / {1 / (1 - percentage_improvement / 100):.2f}x speedup"
            budget_str = f" [Budget: {best_budget}]" if best_budget is not None else ""

            print(f"\nAllowed error ≤ {threshold}: {speedup_str}{budget_str}")

            if per_ev_stats:
                print(f"  Per-eigenvalue error statistics:")
                for idx in sorted(per_ev_stats.keys()):
                    if per_ev_stats[idx]:
                        ev_stats = per_ev_stats[idx]
                        print(f"    Eigenvalue {idx} ({ev_stats['count']} test cases):")
                        print(
                            f"      geomean: {ev_stats['geomean']:.3e} (zeros: {ev_stats.get('zero_count', 0)}/{ev_stats['count']}), median: {ev_stats['median']:.3e}, max: {ev_stats['max']:.3e}"
                        )
                        print(f"      arithmean: {ev_stats['arithmean']:.3e}, std: {ev_stats['std']:.3e}")
                        if ev_stats.get("percentile_under_20pct") is not None:
                            print(f"      {ev_stats['percentile_under_20pct']:.1f}% of errors are < 20% (max > 1.0)")

            if per_evec_stats:
                print(f"  Per-eigenvector error statistics (L2 norm):")
                for idx in sorted(per_evec_stats.keys()):
                    if per_evec_stats[idx]:
                        evec_stats = per_evec_stats[idx]
                        print(f"    Eigenvector {idx} ({evec_stats['count']} test cases):")
                        print(
                            f"      geomean: {evec_stats['geomean']:.3e} (zeros: {evec_stats.get('zero_count', 0)}/{evec_stats['count']}), median: {evec_stats['median']:.3e}, max: {evec_stats['max']:.3e}"
                        )
                        print(f"      arithmean: {evec_stats['arithmean']:.3e}, std: {evec_stats['std']:.3e}")
            elif stats:
                print(f"  Combined error distribution ({stats['count']} test cases):")
                print(
                    f"    median: {stats['median']:.3e}, geomean: {stats['geomean']:.3e} (zeros: {stats.get('zero_count', 0)}/{stats['count']}), arithmean: {stats['arithmean']:.3e}"
                )
                print(f"    std: {stats['std']:.3e}, max: {stats['max']:.3e}")
                if stats.get("percentile_under_20pct") is not None:
                    print(f"    {stats['percentile_under_20pct']:.1f}% of errors are < 20% (max > 1.0)")
            elif worst_error is not None:
                print(f"  Worst case error: {worst_error:.6e}")
        else:
            print(f"\nAllowed error ≤ {threshold}: No data")


def main():
    parser = argparse.ArgumentParser(description="Measure EIG experiments")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze using data from saved file")
    args = parser.parse_args()
    analyze_only = args.analyze_only
    if analyze_only:
        if not os.path.exists("measurements.pkl"):
            print("Measurements file 'measurements.pkl' does not exist.")
            sys.exit(1)
        with open("measurements.pkl", "rb") as f:
            data = pickle.load(f)
        analyze_data(data)
        sys.exit(0)
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    tmp_dir = "tmp"
    if not os.path.exists(tmp_dir):
        print(f"Temporary directory '{tmp_dir}' does not exist.")
        sys.exit(1)
    executables = [f for f in os.listdir(tmp_dir) if f.startswith("eig-fpopt-") and f.endswith(".exe")]
    budgets = []
    executable_paths = []
    for exe in executables:
        m = re.match(r"eig-fpopt-(.*)\.exe", exe)
        if m:
            budget_str = m.group(1)
            try:
                budget = int(budget_str)
                budgets.append(budget)
                executable_paths.append(os.path.join(tmp_dir, exe))
            except ValueError:
                continue
    if not budgets:
        print("No valid executables found.")
        sys.exit(1)
    budgets, executable_paths = zip(*sorted(zip(budgets, executable_paths)))

    original_executable = "./eig.exe"
    original_output_file = os.path.join("outputs", "output_original.txt")
    f32_executable = "./eig-f32.exe"
    f32_output_file = os.path.join("outputs", "output_f32.txt")
    reference_output = "eig_gold.txt"

    if not os.path.exists(reference_output):
        print(f"Reference output file '{reference_output}' does not exist.")
        sys.exit(1)

    if not os.path.exists(original_output_file):
        collect_output(original_executable, original_output_file)

    original_error_result = compute_error_geometric_average(
        reference_output, original_output_file, return_max=True, return_stats=True
    )
    original_per_eigenvalue_stats = compute_error_geometric_average(
        reference_output, original_output_file, return_per_eigenvalue=True
    )

    original_eigenvector_result = compute_error_geometric_average(
        reference_output, original_output_file, compute_eigenvectors=True
    )
    if original_eigenvector_result and original_eigenvector_result != (None, None):
        _, original_per_eigenvector_stats = original_eigenvector_result
    else:
        original_per_eigenvector_stats = None

    if original_error_result is not None and original_error_result != (None, None, None):
        original_error, original_max_error, original_error_stats = original_error_result
        if original_error is not None:
            print(f"Relative error for the original binary: {original_error} (worst case: {original_max_error})")
            if original_error_stats:
                print(f"  Original error distribution ({original_error_stats['count']} test cases):")
                print(
                    f"    median: {original_error_stats['median']:.3e}, geomean: {original_error_stats['geomean']:.3e} (zeros: {original_error_stats.get('zero_count', 0)}/{original_error_stats['count']}), arithmean: {original_error_stats['arithmean']:.3e}"
                )
                print(f"    std: {original_error_stats['std']:.3e}, max: {original_error_stats['max']:.3e}")
                if original_error_stats.get("percentile_under_20pct") is not None:
                    print(f"    {original_error_stats['percentile_under_20pct']:.1f}% of errors are < 20% (max > 1.0)")

            if original_per_eigenvalue_stats:
                print(f"  Original per-eigenvalue errors:")
                for idx in sorted(original_per_eigenvalue_stats.keys()):
                    if original_per_eigenvalue_stats[idx]:
                        stats = original_per_eigenvalue_stats[idx]
                        print(
                            f"    Eigenvalue {idx}: geomean={stats['geomean']:.3e} (zeros: {stats.get('zero_count', 0)}/{stats['count']}), max={stats['max']:.3e}"
                        )

            if original_per_eigenvector_stats:
                print(f"  Original per-eigenvector errors (L2 norm):")
                for idx in sorted(original_per_eigenvector_stats.keys()):
                    if original_per_eigenvector_stats[idx]:
                        stats = original_per_eigenvector_stats[idx]
                        print(
                            f"    Eigenvector {idx}: geomean={stats['geomean']:.3e} (zeros: {stats.get('zero_count', 0)}/{stats['count']}), max={stats['max']:.3e}"
                        )
        else:
            print("Relative error for the original binary could not be computed.")
            original_error = None
            original_error_stats = None
            original_per_eigenvalue_stats = None
            original_per_eigenvector_stats = None
    else:
        print("Relative error for the original binary could not be computed.")
        original_error = None
        original_error_stats = None
        original_per_eigenvalue_stats = None
        original_per_eigenvector_stats = None

    original_runtime = measure_runtime(original_executable, num_runs=NUM_RUNS)

    f32_error = None
    f32_max_error = None
    f32_error_stats = None
    f32_runtime = None

    if os.path.exists(f32_executable):
        print("\n=== Measuring f32 (float) version ===")

        if not os.path.exists(f32_output_file):
            collect_output(f32_executable, f32_output_file)

        f32_error_result = compute_error_geometric_average(
            reference_output, f32_output_file, return_max=True, return_stats=True
        )
        f32_per_eigenvalue_stats = compute_error_geometric_average(
            reference_output, f32_output_file, return_per_eigenvalue=True
        )

        f32_eigenvector_result = compute_error_geometric_average(
            reference_output, f32_output_file, compute_eigenvectors=True
        )
        if f32_eigenvector_result and f32_eigenvector_result != (None, None):
            _, f32_per_eigenvector_stats = f32_eigenvector_result
        else:
            f32_per_eigenvector_stats = None

        if f32_error_result is not None and f32_error_result != (None, None, None):
            f32_error, f32_max_error, f32_error_stats = f32_error_result
            if f32_error is not None:
                print(f"Relative error for f32 binary: {f32_error:.6e} (worst case: {f32_max_error:.6e})")
                if f32_error_stats:
                    print(f"  f32 error distribution ({f32_error_stats['count']} test cases):")
                    print(
                        f"    median: {f32_error_stats['median']:.3e}, geomean: {f32_error_stats['geomean']:.3e} (zeros: {f32_error_stats.get('zero_count', 0)}/{f32_error_stats['count']}), arithmean: {f32_error_stats['arithmean']:.3e}"
                    )
                    print(f"    std: {f32_error_stats['std']:.3e}, max: {f32_error_stats['max']:.3e}")

                if f32_per_eigenvalue_stats:
                    print(f"  f32 per-eigenvalue errors:")
                    for idx in sorted(f32_per_eigenvalue_stats.keys()):
                        if f32_per_eigenvalue_stats[idx]:
                            stats = f32_per_eigenvalue_stats[idx]
                            print(
                                f"    Eigenvalue {idx}: geomean={stats['geomean']:.3e} (zeros: {stats.get('zero_count', 0)}/{stats['count']}), max={stats['max']:.3e}"
                            )

                if f32_per_eigenvector_stats:
                    print(f"  f32 per-eigenvector errors (L2 norm):")
                    for idx in sorted(f32_per_eigenvector_stats.keys()):
                        if f32_per_eigenvector_stats[idx]:
                            stats = f32_per_eigenvector_stats[idx]
                            print(
                                f"    Eigenvector {idx}: geomean={stats['geomean']:.3e} (zeros: {stats.get('zero_count', 0)}/{stats['count']}), max={stats['max']:.3e}"
                            )
            else:
                print("Relative error for f32 binary could not be computed.")
                f32_per_eigenvalue_stats = None
                f32_per_eigenvector_stats = None
        else:
            print("Relative error for f32 binary could not be computed.")
            f32_per_eigenvalue_stats = None
            f32_per_eigenvector_stats = None

        f32_runtime = measure_runtime(f32_executable, num_runs=NUM_RUNS)
        if f32_runtime is not None:
            print(f"f32 runtime: {f32_runtime:.6f} seconds")

            if original_runtime is not None:
                speedup = original_runtime / f32_runtime
                print(f"f32 speedup vs double: {speedup:.2f}x")

            if original_error is not None and f32_error is not None:
                error_ratio = f32_error / original_error
                print(f"f32 error ratio vs double: {error_ratio:.2f}x (higher = worse accuracy)")
    else:
        print(f"\nf32 executable '{f32_executable}' not found. Skipping f32 measurements.")

    print("Starting accuracy measurements...")
    errors = {}
    max_errors = {}
    error_stats = {}
    per_eigenvalue_stats = {}
    per_eigenvector_stats = {}
    with multiprocessing.Pool(processes=min(128, multiprocessing.cpu_count())) as pool:
        results = pool.map(accuracy_task, executable_paths)
        for result in results:
            if result and result[1] is not None:
                budget = result[0]
                errors[budget] = result[1]
                max_errors[budget] = result[2]
                if result[3] is not None:
                    error_stats[budget] = result[3]
                if len(result) > 4 and result[4] is not None:
                    per_eigenvalue_stats[budget] = result[4]
                if len(result) > 5 and result[5] is not None:
                    per_eigenvector_stats[budget] = result[5]

    print("Starting runtime measurements...")
    runtimes = {}
    for exe, budget in tqdm(zip(executable_paths, budgets), total=len(budgets), desc="Measuring runtimes"):
        result = runtime_task(exe)
        if result and result[1] is not None:
            runtimes[budget] = result[1]

    arithmean_errors = {}
    for budget in error_stats:
        if error_stats[budget] and "arithmean" in error_stats[budget]:
            arithmean_errors[budget] = error_stats[budget]["arithmean"]

    data = {
        "budgets": budgets,
        "errors": [errors.get(budget, np.nan) for budget in budgets],
        "arithmean_errors": [arithmean_errors.get(budget, np.nan) for budget in budgets],
        "max_errors": [max_errors.get(budget, np.nan) for budget in budgets],
        "error_stats": error_stats,
        "per_eigenvalue_stats": per_eigenvalue_stats,
        "per_eigenvector_stats": per_eigenvector_stats,
        "runtimes": [runtimes.get(budget, np.nan) for budget in budgets],
        "original_error": original_error if original_error is not None else np.nan,
        "original_error_stats": original_error_stats if original_error_stats is not None else None,
        "original_per_eigenvalue_stats": (
            original_per_eigenvalue_stats if original_per_eigenvalue_stats is not None else None
        ),
        "original_per_eigenvector_stats": (
            original_per_eigenvector_stats if original_per_eigenvector_stats is not None else None
        ),
        "original_runtime": original_runtime if original_runtime is not None else np.nan,
        "f32_error": f32_error if f32_error is not None else np.nan,
        "f32_max_error": f32_max_error if f32_max_error is not None else np.nan,
        "f32_error_stats": f32_error_stats if f32_error_stats is not None else None,
        "f32_per_eigenvalue_stats": f32_per_eigenvalue_stats if f32_per_eigenvalue_stats is not None else None,
        "f32_per_eigenvector_stats": f32_per_eigenvector_stats if f32_per_eigenvector_stats is not None else None,
        "f32_runtime": f32_runtime if f32_runtime is not None else np.nan,
    }

    with open("measurements.pkl", "wb") as f:
        pickle.dump(data, f)
    print("Measurements saved to 'measurements.pkl'")
    analyze_data(data)


if __name__ == "__main__":
    main()
