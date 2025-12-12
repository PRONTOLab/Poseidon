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
import struct


def float_to_ordered_int(x):
    i = struct.unpack("!q", struct.pack("!d", x))[0]
    if i < 0:
        i = 0x8000000000000000 - i
    return i


def ulps_between(a, b):
    ia = float_to_ordered_int(a)
    ib = float_to_ordered_int(b)
    return abs(ib - ia)


NUM_RUNS = 1
SIZE = 27000
ITER_ACCURACY = 1662
ITER_RUNTIME = 100
TIMEOUT_SECONDS = 1000
PROBLEM_SIZE = 50

NUM_PARALLEL = 32


def run_command(
    command, description, capture_output=False, output_file=None, verbose=True, env=None, timeout=TIMEOUT_SECONDS
):
    try:
        if capture_output and output_file:
            with open(output_file, "w") as f:
                subprocess.run(
                    command,
                    stdout=f,
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


def parse_output(file_path):
    """
    Parses the output file to extract:
      - Final Origin Energy
      - MaxAbsDiff, TotalAbsDiff, MaxRelDiff
      - Elapsed time
    """
    with open(file_path, "r") as f:
        content = f.read()

    def extract_value(pattern):
        m = re.search(pattern, content)
        if m:
            s = m.group(1).strip()
            try:
                return float(s)
            except ValueError:
                print(f"Warning: could not convert '{s}' to float for pattern {pattern}")
                return np.nan
        return np.nan

    final_origin_energy = extract_value(r"Final Origin Energy\s*=\s*([-+]?(?:[0-9Ee+\-\.]+|[nN][aA][nN]))")
    max_abs_diff = extract_value(r"MaxAbsDiff\s*=\s*([-+]?(?:[0-9Ee+\-\.]+|[nN][aA][nN]))")
    total_abs_diff = extract_value(r"TotalAbsDiff\s*=\s*([-+]?(?:[0-9Ee+\-\.]+|[nN][aA][nN]))")
    max_rel_diff = extract_value(r"MaxRelDiff\s*=\s*([-+]?(?:[0-9Ee+\-\.]+|[nN][aA][nN]))")
    elapsed_time = extract_value(r"Elapsed time\s*=\s*([-+]?(?:[0-9Ee+\-\.]+|[nN][aA][nN]))")

    return {
        "final_origin_energy": final_origin_energy,
        "max_abs_diff": max_abs_diff,
        "total_abs_diff": total_abs_diff,
        "max_rel_diff": max_rel_diff,
        "elapsed_time": elapsed_time,
    }


def measure_runtime(executable, num_runs):
    """Measure runtime for the executable over multiple runs."""
    print(f"=== Measuring runtime for {executable} ===")
    runtimes = []
    for i in range(num_runs):
        cmd = [executable, "-s", str(PROBLEM_SIZE), "-i", str(ITER_RUNTIME)]
        result = run_command(cmd, f"Running {executable} (run {i+1}/{num_runs})", capture_output=True, verbose=False)
        if result is None:
            runtimes.append(np.nan)
            continue

        runtime_match = re.search(r"Elapsed time\s*=\s*([-+]?(?:[0-9Ee+\-\.]+|[nN][aA][nN]))", result)
        if runtime_match:
            try:
                runtime = float(runtime_match.group(1))
                print(f"Extracted runtime: {runtime:.6f}, iter {i+1}/{num_runs}")
                runtimes.append(runtime)
            except ValueError:
                print(f"Invalid runtime value extracted from output of {executable}")
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
    cmd = [executable, "-s", str(PROBLEM_SIZE), "-i", str(ITER_ACCURACY)]
    run_command(
        cmd, f"Running {executable} and collecting output", capture_output=True, output_file=output_file, verbose=False
    )


def runtime_task(executable):
    """Measure runtime for a single executable."""
    m = re.match(r".*ser-single-forward-fpopt-(.*)\.exe", executable)
    if m:
        try:
            budget = int(m.group(1))
        except ValueError:
            print(f"Could not parse budget from executable name: {executable}")
            return None, None
    else:
        print(f"Could not extract budget from executable name: {executable}")
        return None, None

    runtime = measure_runtime(executable, num_runs=NUM_RUNS)
    if runtime is not None:
        print(f"Measured runtime for budget {budget}: {runtime:.6f}")
    else:
        print(f"Runtime for budget {budget} could not be measured.")
    return budget, runtime


def accuracy_task(executable, ref_values):
    """
    Runs the executable to collect output, parses the output to extract error metrics,
    and computes the ULP error compared to the reference final origin energy.
    Returns the budget and a dictionary of error metrics.
    """
    m = re.match(r".*ser-single-forward-fpopt-(.*)\.exe", executable)
    if m:
        try:
            budget = int(m.group(1))
        except ValueError:
            print(f"Could not parse budget from executable name: {executable}")
            return None, None
    else:
        print(f"Could not extract budget from executable name: {executable}")
        return None, None

    output_file = os.path.join("outputs", f"output_budget_{budget}.txt")
    collect_output(executable, output_file)
    run_vals = parse_output(output_file)

    if np.isnan(run_vals["final_origin_energy"]) or np.isnan(ref_values["final_origin_energy"]):
        final_ulp_error = np.nan
        final_rel_error = np.nan
    else:
        final_ulp_error = ulps_between(run_vals["final_origin_energy"], ref_values["final_origin_energy"])
        if ref_values["final_origin_energy"] != 0:
            final_rel_error = abs(
                (run_vals["final_origin_energy"] - ref_values["final_origin_energy"])
                / ref_values["final_origin_energy"]
            )
        else:
            final_rel_error = np.nan if run_vals["final_origin_energy"] == 0 else np.inf

    errors = {
        "final_ulp_error": final_ulp_error,
        "final_rel_error": final_rel_error,
        "max_abs_diff": run_vals["max_abs_diff"],
        "total_abs_diff": run_vals["total_abs_diff"],
        "max_rel_diff": run_vals["max_rel_diff"],
    }
    os.remove(output_file)
    print(
        f"Budget {budget}: final_origin_energy={run_vals['final_origin_energy']:.16e}, final_ulp_error={final_ulp_error}, "
        f"final_rel_error={final_rel_error:.6e}, "
        f"max_abs_diff={run_vals['max_abs_diff']:.16e}, total_abs_diff={run_vals['total_abs_diff']:.16e}, "
        f"max_rel_diff={run_vals['max_rel_diff']:.16e}"
    )
    return budget, errors


def accuracy_task_wrapper(args):
    executable, ref_values = args
    return accuracy_task(executable, ref_values)


def main():
    parser = argparse.ArgumentParser(description="Measure LULESH experiments")
    parser.add_argument(
        "--sample-percent", type=float, default=10.0, help="Percentage of programs to sample (default: 10%%)"
    )
    parser.add_argument(
        "--accuracy-filter",
        action="store_true",
        help=(
            "Run accuracy on all executables, filter out rel_err >= 1 and, for positive budgets, ULP > 50; "
            "time only the remaining executables and save results to PKL"
        ),
    )
    parser.add_argument(
        "--ulp-threshold", type=float, default=5.0, help="ULP threshold for reporting low-error programs (default: 5)"
    )
    args = parser.parse_args()
    sample_percent = args.sample_percent
    accuracy_filter = args.accuracy_filter
    ulp_threshold = args.ulp_threshold

    gold_file = "lulesh_gold.txt"
    if not os.path.exists(gold_file):
        print(f"Gold file '{gold_file}' does not exist.")
        sys.exit(1)
    ref_values = parse_output(gold_file)
    print(f"Reference final origin energy from gold file: {ref_values['final_origin_energy']:.16e}")

    os.makedirs("outputs", exist_ok=True)

    original_executable = "./ser-single-forward.exe"
    if not os.path.exists(original_executable):
        print(f"Original executable not found at '{original_executable}'.")
        sys.exit(1)

    print("\n=== Measuring original program runtime ===")
    original_runtime = measure_runtime(original_executable, num_runs=NUM_RUNS)
    if original_runtime is None:
        print("Could not measure original runtime. Using 1.0 as placeholder.")
        original_runtime = 1.0

    original_output_file = os.path.join("outputs", "output_original.txt")
    collect_output(original_executable, original_output_file)
    orig_vals = parse_output(original_output_file)
    if np.isnan(orig_vals["final_origin_energy"]) or np.isnan(ref_values["final_origin_energy"]):
        original_ulp_error = np.nan
        original_rel_error = np.nan
    else:
        original_ulp_error = ulps_between(orig_vals["final_origin_energy"], ref_values["final_origin_energy"])
        if ref_values["final_origin_energy"] != 0:
            original_rel_error = abs(
                (orig_vals["final_origin_energy"] - ref_values["final_origin_energy"])
                / ref_values["final_origin_energy"]
            )
        else:
            original_rel_error = np.nan if orig_vals["final_origin_energy"] == 0 else np.inf
    os.remove(original_output_file)
    print(
        f"Original program: final_origin_energy={orig_vals['final_origin_energy']:.16e}, "
        f"final_ulp_error={original_ulp_error}, final_rel_error={original_rel_error:.6e}"
    )

    tmp_dir = "tmp"
    if not os.path.exists(tmp_dir):
        print(f"Temporary directory '{tmp_dir}' does not exist.")
        sys.exit(1)
    executables = [f for f in os.listdir(tmp_dir) if f.startswith("ser-single-forward-fpopt-") and f.endswith(".exe")]
    budgets = []
    executable_paths = []
    for exe in executables:
        m = re.match(r"ser-single-forward-fpopt-(.*)\.exe", exe)
        if m:
            try:
                budget = int(m.group(1))
            except ValueError:
                continue
            budgets.append(budget)
            executable_paths.append(os.path.join(tmp_dir, exe))
    if not budgets:
        print("No valid executables found.")
        sys.exit(1)
    budgets, executable_paths = zip(*sorted(zip(budgets, executable_paths)))

    total_programs = len(budgets)
    if not accuracy_filter:
        sample_size = max(1, int(total_programs * sample_percent / 100))

        print(f"\nTotal programs found: {total_programs}")
        print(f"Sampling {sample_size} programs ({sample_percent}%) uniformly across budget range")

        if sample_size >= total_programs:
            sampled_indices = list(range(total_programs))
        else:
            step = total_programs / sample_size
            sampled_indices = [int(i * step) for i in range(sample_size)]

        sampled_budgets = [budgets[i] for i in sampled_indices]
        sampled_executable_paths = [executable_paths[i] for i in sampled_indices]

        print(f"Selected budgets: {sampled_budgets}")
        print(f"Budget range: {min(sampled_budgets)} to {max(sampled_budgets)}")

        budgets = sampled_budgets
        executable_paths = sampled_executable_paths
    else:
        print(f"\nAccuracy-filter mode: using all {total_programs} executables (no sampling)")

    print("\n=== Starting accuracy measurements ===")
    error_metrics = {}
    low_ulp_programs = []
    cpu_count = NUM_PARALLEL
    with multiprocessing.Pool(processes=cpu_count) as pool:
        args_list = list(zip(executable_paths, [ref_values] * len(executable_paths)))
        results = list(
            tqdm(pool.imap(accuracy_task_wrapper, args_list), total=len(executable_paths), desc="Measuring accuracy")
        )
        for result in results:
            if result and result[1] is not None:
                budget = result[0]
                error_metrics[budget] = result[1]
                ulp_err = result[1].get("final_ulp_error", np.nan)
                if not np.isnan(ulp_err) and ulp_err < ulp_threshold:
                    low_ulp_programs.append((budget, ulp_err))

    if low_ulp_programs:
        print("\n" + "=" * 60)
        print(f"SUMMARY: Programs with ULP error < {ulp_threshold}")
        low_ulp_programs.sort(key=lambda x: x[1])
        for budget, ulp in low_ulp_programs:
            print(f"  Budget {budget}: ULP = {ulp}")
        best_budget, best_ulp = low_ulp_programs[0]
        print(f"\nBest program: Budget {best_budget} with ULP = {best_ulp}")
        print("=" * 60 + "\n")

    if accuracy_filter:
        print("\n=== Filtering executables based on accuracy criteria ===")

        def is_valid_program(budget, metrics):
            if metrics is None:
                return False
            rel_err = metrics.get("final_rel_error", np.nan)
            ulp_err = metrics.get("final_ulp_error", np.nan)
            if np.isnan(rel_err) or rel_err >= 1:
                return False
            if budget > 0 and (np.isnan(ulp_err) or ulp_err > 50):
                return False
            return True

        total_before = len(budgets)
        filtered_pairs = [
            (b, p) for b, p in zip(budgets, executable_paths) if is_valid_program(b, error_metrics.get(b))
        ]
        budgets = [b for b, _ in filtered_pairs]
        executable_paths = [p for _, p in filtered_pairs]
        print(f"Programs passing filters: {len(budgets)} / {total_before}")
        if budgets:
            print(f"Budgets kept: {budgets}")

    print("\n=== Starting runtime measurements for optimized executables ===")
    runtimes = {}
    for exe in tqdm(executable_paths, desc="Measuring runtimes"):
        budget, runtime = runtime_task(exe)
        if budget is not None and runtime is not None:
            runtimes[budget] = runtime

    data = {
        "budgets": budgets,
        "runtimes": [runtimes.get(budget, np.nan) for budget in budgets],
        "final_origin_energy": ref_values["final_origin_energy"],
        "final_ulp_errors": [error_metrics.get(budget, {}).get("final_ulp_error", np.nan) for budget in budgets],
        "final_rel_errors": [error_metrics.get(budget, {}).get("final_rel_error", np.nan) for budget in budgets],
        "max_abs_diff": [error_metrics.get(budget, {}).get("max_abs_diff", np.nan) for budget in budgets],
        "total_abs_diff": [error_metrics.get(budget, {}).get("total_abs_diff", np.nan) for budget in budgets],
        "max_rel_diff": [error_metrics.get(budget, {}).get("max_rel_diff", np.nan) for budget in budgets],
        "original_runtime": original_runtime,
        "original_ulp_error": original_ulp_error,
        "original_rel_error": original_rel_error,
        "original_final_origin_energy": orig_vals.get("final_origin_energy", np.nan),
    }
    with open("measurements.pkl", "wb") as f:
        pickle.dump(data, f)
    print("Measurements saved to 'measurements.pkl'")

    if accuracy_filter:
        print("Accuracy-filter mode completed. Exiting after saving PKL.")
        return

    print("Measurements completed and saved to measurements.pkl")


if __name__ == "__main__":
    main()
