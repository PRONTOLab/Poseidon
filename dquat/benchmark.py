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

NUM_PARALLEL = 32


def geomean(values):
    assert len(values) > 0, "Cannot compute geometric mean of an empty list"
    sum_log = 0.0
    nonzero_count = 0

    for x in values:
        if x != 0:
            sum_log += math.log(x)
            nonzero_count += 1

    if nonzero_count == 0:
        return 0.0

    return math.exp(sum_log / nonzero_count)


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


def compute_error(reference_output, test_output, max_iterations):
    ref_matrices_dict = extract_all_matrices(reference_output, max_iterations)
    test_matrices_dict = extract_all_matrices(test_output, max_iterations)

    if ref_matrices_dict is None or test_matrices_dict is None:
        print("Could not extract matrices for error computation.")
        return None

    if set(ref_matrices_dict.keys()) != set(test_matrices_dict.keys()):
        print("Mismatch in iteration numbers between reference and test outputs.")
        return None

    all_relative_diffs = []

    ref_norm = {}
    for iteration in ref_matrices_dict:
        if iteration > max_iterations:
            break
        ref_matrix = ref_matrices_dict[iteration]
        ref_norm[iteration] = np.linalg.norm(ref_matrix, ord="fro")

    for iteration in ref_matrices_dict:
        if iteration > max_iterations:
            break
        ref_matrix = ref_matrices_dict[iteration]
        test_matrix = test_matrices_dict.get(iteration, None)

        if test_matrix is None:
            print(f"Missing matrix for iteration {iteration}. Skipping.")
            continue

        if ref_matrix.shape != test_matrix.shape:
            print(f"Mismatch in matrix shape at iteration {iteration}. Skipping.")
            continue

        norm_ref = ref_norm.get(iteration, None)
        if norm_ref is None or norm_ref == 0:
            print(f"Reference matrix norm is zero or undefined for iteration {iteration}. Skipping.")
            continue

        diff = ref_matrix - test_matrix
        fro_diff = np.linalg.norm(diff, ord="fro")
        relative_diff = fro_diff / norm_ref
        all_relative_diffs.append(relative_diff)

    if not all_relative_diffs:
        print("No valid relative differences to compute error.")
        return np.nan, np.nan

    try:
        relative_error = geomean(all_relative_diffs)
        max_error = max(all_relative_diffs)
        return relative_error, max_error
    except (ValueError, OverflowError) as e:
        print(f"Error computing geometric average: {e}")
        return np.nan, np.nan


def extract_all_matrices(output_file, max_iterations):
    matrices_dict = {}
    current_iteration = None
    iterations_processed = 0

    with open(output_file, "r") as f:
        lines = f.readlines()

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        iter_match = re.match(r"Test\s+(\d+):", line)
        if iter_match:
            current_iteration = int(iter_match.group(1))
            matrices_dict[current_iteration] = []
            iterations_processed += 1
            if max_iterations is not None and iterations_processed > max_iterations:
                break
            idx += 1
            continue

        if current_iteration is not None:
            if "dquat_dexpmap ^ T value:" in line:
                if idx + 4 >= len(lines):
                    print(f"Insufficient lines for matrix at iteration {current_iteration} in {output_file}")
                    break
                matrix = []
                for j in range(1, 4):
                    vec_line = lines[idx + j].strip()
                    vec = []
                    for val_str in vec_line.split():
                        if val_str.lower() in ["nan", "-nan"]:
                            vec.append(np.nan)
                        else:
                            try:
                                vec.append(float(val_str))
                            except ValueError:
                                vec.append(np.nan)
                    if len(vec) != 4:
                        print(f"Invalid matrix row length at iteration {current_iteration} in {output_file}")
                        vec = [np.nan] * 4
                    matrix.append(vec)
                matrix = np.array(matrix)
                matrices_dict[current_iteration] = matrix
                idx += 4
                continue
        idx += 1
    if not matrices_dict:
        print(f"No matrices found in {output_file}")
        return None
    return matrices_dict


def accuracy_task(executable, max_iterations, reference_output="dquat_gold.txt"):
    m = re.match(r".*dquat-fpopt-(.*)\.exe", executable)
    if m:
        budget = int(m.group(1))
    else:
        print(f"Could not extract budget from executable name: {executable}")
        return None, None
    output_file = os.path.join("outputs", f"output_budget_{budget}.txt")
    collect_output(executable, output_file)
    error, max_error = compute_error(reference_output, output_file, max_iterations)
    if error is not None:
        try:
            print(f"Relative error for budget {budget}: geomean={error:.6e} (worst case: {max_error:.6e})")
        except Exception:
            print(f"Relative error for budget {budget}: geomean={error} (worst case: {max_error})")
    else:
        print(f"Relative error for budget {budget} could not be computed.")

    os.remove(output_file)
    return budget, error, max_error


def runtime_task(executable, reference_output="dquat_gold.txt"):
    m = re.match(r".*dquat-fpopt-(.*)\.exe", executable)
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


def analyze_data(data, thresholds=None):
    budgets = data["budgets"]
    runtimes = data["runtimes"]
    errors = data["errors"]
    original_runtime = data["original_runtime"]
    original_error = data["original_error"]

    if thresholds is None:
        thresholds = [0, 1e-14, 1e-12, 1e-10, 1e-9, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 0.2, 0.3, 0.4, 0.5, 0.9, 1]

    min_budget = None
    for budget, error in zip(budgets, errors):
        if np.isnan(error):
            if min_budget is None or budget < min_budget:
                min_budget = budget

    print("Minimum budget with NaN error:", min_budget)

    min_runtime_ratios = {}
    min_errors = {}
    for threshold in thresholds:
        min_ratio = None
        min_error = None
        for err, runtime in zip(errors, runtimes):
            if err is not None and runtime is not None and err <= threshold:
                runtime_ratio = runtime / original_runtime
                if min_ratio is None or runtime_ratio < min_ratio:
                    min_ratio = runtime_ratio
                    min_error = err
        if min_ratio is not None:
            min_runtime_ratios[threshold] = min_ratio
            min_errors[threshold] = min_error

    overall_runtime_improvements = {}
    for threshold in thresholds:
        ratio = min_runtime_ratios.get(threshold)
        if ratio is not None:
            percentage_improvement = (1 - ratio) * 100
            overall_runtime_improvements[threshold] = percentage_improvement
        else:
            overall_runtime_improvements[threshold] = None

    print("\nPercentage of runtime improvements while allowing some level of relative error:")
    for threshold in thresholds:
        percentage_improvement = overall_runtime_improvements[threshold]
        if percentage_improvement is not None:
            print(
                f"Allowed relative error ≤ {threshold}: {percentage_improvement:.2f}% runtime reduction / {1 / (1 - percentage_improvement / 100):.2f}x speedup; relative error = {min_errors[threshold]}"
            )
        else:
            print(f"Allowed relative error ≤ {threshold}: No data")


def main():
    parser = argparse.ArgumentParser(description="Measure dquat experiments and analyze results")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze using data from saved file")
    parser.add_argument(
        "--max-iterations", type=int, default=1000, help="Maximum number of iterations to check for accuracy"
    )
    args = parser.parse_args()
    analyze_only = args.analyze_only
    max_iterations = args.max_iterations

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
    executables = [f for f in os.listdir(tmp_dir) if f.startswith("dquat-fpopt-") and f.endswith(".exe")]
    budgets = []
    executable_paths = []
    for exe in executables:
        m = re.match(r"dquat-fpopt-(.*)\.exe", exe)
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

    original_executable = "./dquat.exe"
    original_output_file = os.path.join("outputs", "output_original.txt")
    reference_output = "dquat_gold.txt"

    if not os.path.exists(reference_output):
        print(f"Reference output file '{reference_output}' does not exist.")
        sys.exit(1)

    if not os.path.exists(original_output_file):
        collect_output(original_executable, original_output_file)

    original_error, original_max_error = compute_error(reference_output, original_output_file, max_iterations)
    if original_error is not None:
        try:
            print(
                f"Relative error for the original binary: {original_error:.6e} (worst case: {original_max_error:.6e})"
            )
        except Exception:
            print(f"Relative error for the original binary: {original_error} (worst case: {original_max_error})")
    else:
        print("Relative error for the original binary could not be computed.")

    original_runtime = measure_runtime(original_executable, num_runs=NUM_RUNS)

    print("Starting accuracy measurements...")
    errors = {}
    max_errors = {}
    with multiprocessing.Pool(NUM_PARALLEL) as pool:
        results = pool.starmap(accuracy_task, [(exe, max_iterations) for exe in executable_paths])
        for result in results:
            if result and result[1] is not None and not np.isnan(result[1]):
                budget = result[0]
                errors[budget] = result[1]
                max_errors[budget] = result[2]
            else:
                try:
                    print(f"Excluding optimized program for budget {result[0]} due to NaN output.")
                except Exception:
                    print("Excluding optimized program due to NaN output.")

    print("Starting runtime measurements...")
    runtimes = {}
    for exe, budget in tqdm(zip(executable_paths, budgets), total=len(budgets), desc="Measuring runtimes"):
        if budget not in errors:
            continue
        result = runtime_task(exe)
        if result and result[1] is not None:
            runtimes[budget] = result[1]

    data = {
        "budgets": budgets,
        "errors": [errors.get(budget, np.nan) for budget in budgets],
        "max_errors": [max_errors.get(budget, np.nan) for budget in budgets],
        "runtimes": [runtimes.get(budget, np.nan) for budget in budgets],
        "original_error": original_error if original_error is not None else np.nan,
        "original_max_error": original_max_error if original_max_error is not None else np.nan,
        "original_runtime": original_runtime if original_runtime is not None else np.nan,
    }

    valid_indices = [i for i, err in enumerate(data["errors"]) if not np.isnan(err)]
    excluded_count = len(data["errors"]) - len(valid_indices)
    if excluded_count > 0:
        print(f"Excluded {excluded_count} optimized program(s) due to NaN outputs.")

    data["budgets"] = [data["budgets"][i] for i in valid_indices]
    data["errors"] = [data["errors"][i] for i in valid_indices]
    data["max_errors"] = [data["max_errors"][i] for i in valid_indices]
    data["runtimes"] = [data["runtimes"][i] for i in valid_indices]

    with open("measurements.pkl", "wb") as f:
        pickle.dump(data, f)
    print("Measurements saved to 'measurements.pkl'")
    analyze_data(data)


if __name__ == "__main__":
    main()
