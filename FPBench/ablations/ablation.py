#!/usr/bin/env python3

import os
import subprocess
import sys
import shutil
import re
import argparse
import matplotlib.pyplot as plt
import math
import random
import numpy as np
import json
from statistics import mean
import pickle
from tqdm import trange
from matplotlib import rcParams
from concurrent.futures import ProcessPoolExecutor, as_completed
import seaborn as sns

HOME = "/root"
ENZYME_PATH = os.path.join(HOME, "Poseidon/Enzyme/build/Enzyme/ClangEnzyme-22.so")
LLVM_PATH = os.path.join(HOME, "Poseidon/llvm-project/build/bin")
CXX = os.path.join(LLVM_PATH, "clang++")

CXXFLAGS = [
    "-O3",
    "-I" + os.path.join(HOME, "include"),
    "-L" + os.path.join(HOME, "lib"),
    "-I" + os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "-I/usr/include/c++/13",
    "-I/usr/include/x86_64-linux-gnu/c++/13",
    "-L/usr/lib/gcc/x86_64-linux-gnu/13",
    "-fno-exceptions",
    f"-fpass-plugin={ENZYME_PATH}",
    "-Xclang",
    "-load",
    "-Xclang",
    ENZYME_PATH,
    "-lmpfr",
    "-ffast-math",
    "-march=native",
]

FPOPTFLAGS_BASE_TEMPLATE = [
    "-mllvm",
    "--fpopt-enable-herbie",
    "-mllvm",
    "--fpopt-print",
    "-mllvm",
    "--fpopt-enable-solver",
    "-mllvm",
    "--fpopt-enable-pt",
    "-mllvm",
    "--fpopt-early-prune",
    "-mllvm",
    "--fpopt-show-pt-details",
    "-mllvm",
    "--fpopt-show-table",
    "-mllvm",
    "--fpopt-comp-cost-budget=0",
    "-mllvm",
    "--herbie-num-threads=8",
    "-mllvm",
    "--herbie-timeout=1000",
    "-mllvm",
    "--fpopt-num-samples=1024",
    "-mllvm",
    f"--fpopt-cost-model-path={HOME}/Poseidon/cost-model/cm.csv",
    "-mllvm",
    "--fpopt-cache-path=cache",
    "-mllvm",
    "--fpopt-min-uses-split=99",
    "-mllvm",
    "--fpopt-min-ops-split=99",
]

SRC = "example.c"
NUM_RUNS = 10
DRIVER_NUM_SAMPLES = 10000000
MAX_TESTED_COSTS = 999


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


def run_command(command, description, capture_output=False, output_file=None, verbose=True, timeout=None):
    print(f"=== {description} ===")
    print("Running:", " ".join(command))
    try:
        if capture_output and output_file:
            with open(output_file, "w") as f:
                subprocess.check_call(command, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
        elif capture_output:
            result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=timeout)
            return result.stdout
        else:
            if verbose:
                subprocess.check_call(command, timeout=timeout)
            else:
                subprocess.check_call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"Command '{' '.join(command)}' timed out after {timeout} seconds.")
        return
    except subprocess.CalledProcessError as e:
        print(f"Error during: {description}")
        if capture_output and output_file:
            print(f"Check the output file: {output_file} for details.")
        else:
            print(e)
        sys.exit(e.returncode)


def clean_tmp_except_pkl(tmp_dir):
    for entry in os.listdir(tmp_dir):
        full_path = os.path.join(tmp_dir, entry)
        if os.path.isfile(full_path) and not full_path.endswith(".pkl"):
            os.remove(full_path)
            print(f"Removed file: {full_path}")
        elif os.path.isdir(full_path):
            shutil.rmtree(full_path)
            print(f"Removed directory: {full_path}")


def generate_example_cpp(tmp_dir, original_prefix, prefix):
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fpopt-original-driver-generator.py")
    print(f"=== Running {script} ===")
    src_prefixed = os.path.join(tmp_dir, f"{original_prefix}{SRC}")
    dest_prefixed = os.path.join(tmp_dir, f"{prefix}example.cpp")
    run_command(
        ["python3", script, src_prefixed, dest_prefixed, "example", str(DRIVER_NUM_SAMPLES)],
        f"Generating example.cpp from {SRC}",
    )
    if not os.path.exists(dest_prefixed):
        print(f"Failed to generate {dest_prefixed}.")
        sys.exit(1)
    print(f"Generated {dest_prefixed} successfully.")


def compile_example_exe(tmp_dir, prefix):
    source = os.path.join(tmp_dir, f"{prefix}example.cpp")
    output = os.path.join(tmp_dir, f"{prefix}example.exe")
    cmd = [CXX, source] + CXXFLAGS + ["-o", output]
    run_command(cmd, f"Compiling {output}")


def compile_example_logged_exe(tmp_dir, prefix):
    source = os.path.join(tmp_dir, f"{prefix}example.cpp")
    output = os.path.join(tmp_dir, f"{prefix}example-logged.exe")
    cmd = (
        [CXX, source]
        + CXXFLAGS
        + [
            "-mllvm",
            "--fpprofile-generate",
            "-L",
            os.path.join(HOME, "Poseidon/Enzyme/build/Enzyme"),
            "-lEnzymeFPProfile",
            "-o",
            output,
        ]
    )
    run_command(cmd, f"Compiling {output} with fpprofile-generate")


def generate_fpprofile(tmp_dir, prefix):
    exe = os.path.join(tmp_dir, f"{prefix}example-logged.exe")
    profile_dir = os.path.join(tmp_dir, f"{prefix}fpprofile")

    if not os.path.exists(exe):
        print(f"Executable {exe} not found. Cannot generate fpprofile.")
        sys.exit(1)

    print(f"=== Running {exe} with --prof to generate fpprofile ===")

    if os.path.exists("fpprofile"):
        shutil.rmtree("fpprofile")
    if os.path.exists(profile_dir):
        shutil.rmtree(profile_dir)

    try:
        env = os.environ.copy()
        subprocess.check_call([exe, "--prof"], timeout=300, env=env)

        if os.path.exists("fpprofile"):
            shutil.move("fpprofile", profile_dir)
            print(f"FPProfile generated and moved to {profile_dir}")
        else:
            print(f"Error: fpprofile directory was not created")
            sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f"Execution of {exe} timed out.")
        if os.path.exists(exe):
            os.remove(exe)
            print(f"Removed executable {exe} due to timeout.")
        return
    except subprocess.CalledProcessError as e:
        print(f"Error running {exe} with --prof")
        sys.exit(e.returncode)


def compile_example_fpopt_exe(tmp_dir, prefix, fpoptflags, output="example-fpopt.exe", verbose=True):
    source = os.path.join(tmp_dir, f"{prefix}example.cpp")
    output_path = os.path.join(tmp_dir, f"{prefix}{output}")
    profile_dir = os.path.join(tmp_dir, f"{prefix}fpprofile")

    updated_fpoptflags = []
    for flag in fpoptflags:
        if flag.startswith("--fpprofile-use="):
            updated_fpoptflags.append(f"--fpprofile-use={profile_dir}")
        else:
            updated_fpoptflags.append(flag)

    cmd = [CXX, source] + CXXFLAGS + updated_fpoptflags + ["-o", output_path]
    log_path = os.path.join("logs", f"{prefix}compile_fpopt.log")
    if output == "example-fpopt.exe":
        run_command(
            cmd,
            f"Compiling {output_path} with FPOPTFLAGS",
            capture_output=True,
            output_file=log_path,
            verbose=verbose,
        )
    else:
        run_command(
            cmd,
            f"Compiling {output_path} with FPOPTFLAGS",
            verbose=verbose,
        )


def parse_critical_comp_costs(tmp_dir, prefix):
    budgets_file = os.path.join("cache", "budgets.txt")
    print(f"=== Reading critical computation costs from {budgets_file} ===")
    if not os.path.exists(budgets_file):
        print(f"Budgets file {budgets_file} does not exist.")
        sys.exit(1)
    with open(budgets_file, "r") as f:
        content = f.read().strip()
    if not content:
        print(f"Budgets file {budgets_file} is empty.")
        sys.exit(1)
    try:
        costs = [int(cost.strip()) for cost in content.split(",") if cost.strip() != ""]
    except ValueError as e:
        print(f"Error parsing budgets from file {budgets_file}: {e}")
        sys.exit(1)
    print(f"Read computation costs: {costs}")
    if not costs:
        print("No valid computation costs found in budgets.txt.")
        sys.exit(1)
    num_to_sample = min(MAX_TESTED_COSTS, len(costs))
    sampled_costs = random.sample(costs, num_to_sample)
    sampled_costs_sorted = sorted(sampled_costs)
    print(f"Sampled computation costs (sorted): {sampled_costs_sorted}")
    return sampled_costs_sorted


def measure_runtime(tmp_dir, prefix, executable, num_runs=NUM_RUNS):
    print(f"=== Measuring runtime for {executable} ===")
    runtimes = []
    exe_path = os.path.join(tmp_dir, f"{prefix}{executable}")
    for i in trange(1, num_runs + 1):
        try:
            result = subprocess.run([exe_path], capture_output=True, text=True, check=True, timeout=300)
            output = result.stdout
            match = re.search(r"Total runtime: ([\d\.]+) seconds", output)
            if match:
                runtime = float(match.group(1))
                runtimes.append(runtime)
            else:
                print(f"Could not parse runtime from output on run {i}")
                sys.exit(1)
        except subprocess.TimeoutExpired:
            print(f"Execution of {exe_path} timed out on run {i}")
            if os.path.exists(exe_path):
                os.remove(exe_path)
                print(f"Removed executable {exe_path} due to timeout.")
            return None
        except subprocess.CalledProcessError as e:
            print(f"Error running {exe_path} on run {i}")
            sys.exit(e.returncode)
    if runtimes:
        average_runtime = mean(runtimes)
        print(f"Average runtime for {prefix}{executable}: {average_runtime:.6f} seconds")
        return average_runtime
    else:
        print(f"No successful runs for {prefix}{executable}")
        return None


def get_values_file_path(tmp_dir, prefix, binary_name):
    return os.path.join(tmp_dir, f"{prefix}{binary_name}-values.txt")


def generate_example_values(tmp_dir, prefix):
    binary_name = "example.exe"
    exe = os.path.join(tmp_dir, f"{prefix}{binary_name}")
    output_values_file = get_values_file_path(tmp_dir, prefix, binary_name)
    cmd = [exe, "--output-path", output_values_file]
    run_command(cmd, f"Generating function values from {binary_name}", verbose=False, timeout=300)


def generate_values(tmp_dir, prefix, binary_name):
    exe = os.path.join(tmp_dir, f"{prefix}{binary_name}")
    values_file = get_values_file_path(tmp_dir, prefix, binary_name)
    cmd = [exe, "--output-path", values_file]
    run_command(cmd, f"Generating function values from {binary_name}", verbose=False, timeout=300)


def compile_golden_exe(tmp_dir, prefix):
    source = os.path.join(tmp_dir, f"{prefix}golden.cpp")
    output = os.path.join(tmp_dir, f"{prefix}golden.exe")
    cmd = [CXX, source] + CXXFLAGS + ["-o", output]
    run_command(cmd, f"Compiling {output}")


def generate_golden_values(tmp_dir, original_prefix, prefix):
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fpopt-golden-driver-generator.py")
    src_prefixed = os.path.join(tmp_dir, f"{original_prefix}{SRC}")
    dest_prefixed = os.path.join(tmp_dir, f"{prefix}golden.cpp")
    cur_prec = 128
    max_prec = 4096
    PREC_step = 128
    prev_output = None
    output_values_file = get_values_file_path(tmp_dir, prefix, "golden.exe")
    while cur_prec <= max_prec:
        run_command(
            ["python3", script, src_prefixed, dest_prefixed, str(cur_prec), "example", str(DRIVER_NUM_SAMPLES)],
            f"Generating golden.cpp with PREC={cur_prec}",
        )
        if not os.path.exists(dest_prefixed):
            print(f"Failed to generate {dest_prefixed}.")
            sys.exit(1)
        print(f"Generated {dest_prefixed} successfully.")

        compile_golden_exe(tmp_dir, prefix)

        exe = os.path.join(tmp_dir, f"{prefix}golden.exe")
        cmd = [exe, "--output-path", output_values_file]
        run_command(cmd, f"Generating golden values with PREC={cur_prec}", verbose=False)

        if not os.path.exists(output_values_file):
            print(f"Failed to generate golden values at PREC={cur_prec} due to timeout.")
            return

        with open(output_values_file, "r") as f:
            output = f.read()

        if output == prev_output:
            print(f"Golden values converged at PREC={cur_prec}")
            break
        else:
            prev_output = output
            cur_prec += PREC_step
    else:
        print(f"Failed to converge golden values up to PREC={max_prec}")
        sys.exit(1)


def get_avg_rel_error(tmp_dir, prefix, golden_values_file, binaries):
    with open(golden_values_file, "r") as f:
        golden_values = [float(line.strip()) for line in f]

    errors = {}
    for binary in binaries:
        values_file = get_values_file_path(tmp_dir, prefix, binary)
        if not os.path.exists(values_file):
            print(f"Values file {values_file} does not exist. Skipping error calculation for {binary}.")
            errors[binary] = None
            continue
        with open(values_file, "r") as f:
            values = [float(line.strip()) for line in f]
        if len(values) != len(golden_values):
            print(f"Number of values in {values_file} does not match golden values")
            sys.exit(1)

        valid_errors = []
        for v, g in zip(values, golden_values):
            if math.isnan(v) or math.isnan(g):
                continue
            if g == 0:
                continue
            error = max(abs((v - g) / g), abs(math.ulp(g) / g))
            valid_errors.append(error)

        if not valid_errors:
            print(f"No valid data to compute rel error for binary {binary}. Setting rel error to None.")
            errors[binary] = None
            continue

        try:
            errors[binary] = geomean(valid_errors)
        except OverflowError:
            print(
                f"Overflow error encountered while computing geometric mean for binary {binary}. Setting rel error to None."
            )
            errors[binary] = None
        except ZeroDivisionError:
            print(f"No valid errors to compute geometric mean for binary {binary}. Setting rel error to None.")
            errors[binary] = None

    return errors


def build_all(tmp_dir, logs_dir, original_prefix, prefix, fpoptflags):
    generate_example_cpp(tmp_dir, original_prefix, prefix)
    compile_example_exe(tmp_dir, prefix)
    compile_example_logged_exe(tmp_dir, prefix)
    generate_fpprofile(tmp_dir, prefix)
    fpoptflags_with_profile = fpoptflags.copy()
    fpoptflags_with_profile.append("-mllvm")
    fpoptflags_with_profile.append(f"--fpprofile-use={os.path.join(tmp_dir, f'{prefix}fpprofile')}")
    compile_example_fpopt_exe(tmp_dir, prefix, fpoptflags_with_profile, output="example-fpopt.exe")
    print("=== Initial build process completed successfully ===")


def process_cost(args):
    cost, tmp_dir, prefix, fpoptflags = args

    print(f"\n=== Processing computation cost budget: {cost} ===")
    fpoptflags_cost = []
    for flag in fpoptflags:
        if flag.startswith("--fpopt-comp-cost-budget="):
            fpoptflags_cost.append(f"--fpopt-comp-cost-budget={cost}")
        else:
            fpoptflags_cost.append(flag)

    fpoptflags_cost.append("-mllvm")
    fpoptflags_cost.append(f"--fpprofile-use={os.path.join(tmp_dir, f'{prefix}fpprofile')}")

    output_binary = f"example-fpopt-{cost}.exe"

    compile_example_fpopt_exe(tmp_dir, prefix, fpoptflags_cost, output=output_binary, verbose=False)
    generate_values(tmp_dir, prefix, output_binary)

    return cost, output_binary


def benchmark(tmp_dir, logs_dir, original_prefix, prefix, plots_dir, fpoptflags, num_parallel=1):
    costs = parse_critical_comp_costs(tmp_dir, prefix)

    original_avg_runtime = measure_runtime(tmp_dir, prefix, "example.exe", NUM_RUNS)
    original_runtime = original_avg_runtime

    if original_runtime is None:
        print("Original binary timed out. Proceeding as if it doesn't exist.")
        return

    generate_example_values(tmp_dir, prefix)

    generate_golden_values(tmp_dir, original_prefix, prefix)

    golden_values_file = get_values_file_path(tmp_dir, prefix, "golden.exe")
    example_binary = "example.exe"
    rel_errs_example = get_avg_rel_error(tmp_dir, prefix, golden_values_file, [example_binary])
    rel_err_example = rel_errs_example[example_binary]
    print(f"Average Rel Error for {prefix}example.exe: {rel_err_example}")

    data_tuples = []

    args_list = [(cost, tmp_dir, prefix, fpoptflags) for cost in costs]

    if num_parallel == 1:
        for args in args_list:
            cost, output_binary = process_cost(args)
            data_tuples.append((cost, output_binary))
    else:
        with ProcessPoolExecutor(max_workers=num_parallel) as executor:
            future_to_cost = {executor.submit(process_cost, args): args[0] for args in args_list}
            for future in as_completed(future_to_cost):
                cost = future_to_cost[future]
                try:
                    cost_result, output_binary = future.result()
                    data_tuples.append((cost_result, output_binary))
                except Exception as exc:
                    print(f"Cost {cost} generated an exception: {exc}")

    data_tuples_sorted = sorted(data_tuples, key=lambda x: x[0])
    sorted_budgets, sorted_optimized_binaries = zip(*data_tuples_sorted) if data_tuples_sorted else ([], [])

    sorted_runtimes = []
    for cost, output_binary in zip(sorted_budgets, sorted_optimized_binaries):
        avg_runtime = measure_runtime(tmp_dir, prefix, output_binary, NUM_RUNS)
        if avg_runtime is not None:
            sorted_runtimes.append(avg_runtime)
        else:
            print(f"Skipping cost {cost} due to runtime measurement failure.")
            sorted_runtimes.append(None)

    errors_dict = get_avg_rel_error(tmp_dir, prefix, golden_values_file, sorted_optimized_binaries)
    sorted_errors = []
    for binary in sorted_optimized_binaries:
        sorted_errors.append(errors_dict.get(binary))
        print(f"Average rel error for {binary}: {errors_dict.get(binary)}")

    sorted_budgets = list(sorted_budgets)
    sorted_runtimes = list(sorted_runtimes)
    sorted_errors = list(sorted_errors)

    data = {
        "budgets": sorted_budgets,
        "runtimes": sorted_runtimes,
        "errors": sorted_errors,
        "original_runtime": original_runtime,
        "original_error": rel_err_example,
    }

    table_json_path = os.path.join("cache", "table.json")
    if os.path.exists(table_json_path):
        with open(table_json_path, "r") as f:
            table_data = json.load(f)
        if "costToAccuracyMap" in table_data:
            predicted_costs = []
            predicted_errors = []
            for k, v in table_data["costToAccuracyMap"].items():
                try:
                    cost_val = float(k)
                    err_val = float(v)
                    predicted_costs.append(cost_val)
                    predicted_errors.append(err_val)
                except ValueError:
                    pass
            pred_sorted_indices = np.argsort(predicted_costs)
            predicted_costs = np.array(predicted_costs)[pred_sorted_indices].tolist()
            predicted_errors = np.array(predicted_errors)[pred_sorted_indices].tolist()
            data["predicted_data"] = {"costs": predicted_costs, "errors": predicted_errors}

    data_file = os.path.join(tmp_dir, f"{prefix}benchmark_data.pkl")
    with open(data_file, "wb") as f:
        pickle.dump(data, f)
    print(f"Benchmark data saved to {data_file}")

    return data


def build_with_benchmark(tmp_dir, logs_dir, plots_dir, original_prefix, prefix, fpoptflags, num_parallel=1):
    build_all(tmp_dir, logs_dir, original_prefix, prefix, fpoptflags)
    data = benchmark(tmp_dir, logs_dir, original_prefix, prefix, plots_dir, fpoptflags, num_parallel)
    return data


def remove_cache_dir():
    cache_dir = "cache"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print("=== Removed existing cache directory ===")


def plot_ablation_results_cost_model(tmp_dir, plots_dir, prefix, output_format="png", show_prediction=False):
    ablation_data_file = os.path.join(tmp_dir, f"{prefix}ablation-cost-model.pkl")
    if not os.path.exists(ablation_data_file):
        print(f"Ablation data file {ablation_data_file} does not exist. Cannot plot.")
        sys.exit(1)
    with open(ablation_data_file, "rb") as f:
        all_data = pickle.load(f)

    if not all_data:
        print("No data to plot.")
        sys.exit(1)

    sns.set_theme(style="whitegrid")
    rcParams["axes.labelsize"] = 24
    rcParams["xtick.labelsize"] = 20
    rcParams["ytick.labelsize"] = 18
    rcParams["legend.fontsize"] = 20
    blue = "#2287E6"
    yellow = "#FFBD59"
    red = "#FF6666"

    if show_prediction and any("predicted_data" in d for d in all_data.values()):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        ax2 = None

    first_key = next(iter(all_data))
    original_runtime = all_data[first_key]["original_runtime"]
    original_error = all_data[first_key]["original_error"]

    colors = [blue, yellow]
    labels = ["Instruction-JIT", "TTI"]

    for idx, key in enumerate(["custom", "tti"]):
        data = all_data[key]
        runtimes = data["runtimes"]
        errors = data["errors"]

        data_points = list(zip(runtimes, errors))
        filtered_data = [(r, e) for r, e in data_points if r is not None and e is not None]
        if not filtered_data:
            print(f"No valid data to plot for {labels[idx]}.")
            continue
        runtimes_filtered, errors_filtered = zip(*filtered_data)
        color = colors[idx]
        label = labels[idx]
        marker = "*"

        ax1.scatter(runtimes_filtered, errors_filtered, label=label, color=color, marker=marker, s=200)
        points = np.array(filtered_data)
        sorted_indices = np.argsort(points[:, 0])
        sorted_points = points[sorted_indices]

        pareto_front = [sorted_points[0]]
        for point in sorted_points[1:]:
            if point[1] < pareto_front[-1][1]:
                pareto_front.append(point)
        pareto_front = np.array(pareto_front)
        ax1.step(
            pareto_front[:, 0],
            pareto_front[:, 1],
            where="post",
            linestyle="--",
            color=color,
        )

        if show_prediction and "predicted_data" in data and ax2 is not None:
            p_costs = data["predicted_data"]["costs"]
            p_errors = data["predicted_data"]["errors"]
            if p_costs and p_errors:
                ax2.scatter(p_costs, p_errors, label=label, color=color, marker=marker)
                pred_points = np.column_stack((p_costs, p_errors))
                pred_sorted_indices = np.argsort(pred_points[:, 0])
                pred_sorted_points = pred_points[pred_sorted_indices]
                pred_pareto = [pred_sorted_points[0]]
                for pt in pred_sorted_points[1:]:
                    if pt[1] < pred_pareto[-1][1]:
                        pred_pareto.append(pt)
                pred_pareto = np.array(pred_pareto)
                ax2.step(
                    pred_pareto[:, 0],
                    pred_pareto[:, 1],
                    where="post",
                    linestyle="--",
                    color=color,
                )

    ax1.scatter(original_runtime, original_error, marker="^", color=red, s=200, label="Baseline")
    ax1.set_xlabel("Runtime (second)")
    ax1.set_ylabel("Relative Error")
    ax1.set_yscale("symlog", linthresh=1e-14)
    ax1.set_ylim(bottom=-1e-14)
    ax1.legend()
    ax1.grid(True)

    if ax2 is not None:
        ax2.set_xlabel("Cost Budget")
        ax2.set_ylabel("Predicted Error")
        ax2.set_yscale("symlog", linthresh=1e-14)
        ax2.set_ylim(bottom=-1e-14)
        ax2.legend()
        ax2.grid(True)

    plot_filename = os.path.join(plots_dir, f"{prefix}ablation.{output_format}")
    plt.savefig(plot_filename, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Ablation plot saved to {plot_filename}")


def remove_mllvm_flag(flags_list, flag_prefix):
    new_flags = []
    i = 0
    while i < len(flags_list):
        if flags_list[i] == "-mllvm" and i + 1 < len(flags_list) and flags_list[i + 1].startswith(flag_prefix):
            i += 2
        else:
            new_flags.append(flags_list[i])
            i += 1
    return new_flags


def extract_function_from_exported(prefix, tmp_dir):
    parts = prefix.rstrip("-").split("-")
    if len(parts) < 2:
        print(f"Cannot extract benchmark and function from prefix: {prefix}")
        return False

    func_name = parts[-1]
    benchmark_name = "-".join(parts[:-1])

    exported_dir = "../exported"
    exported_file = os.path.join(exported_dir, f"{benchmark_name}.fpcore.c")

    if not os.path.exists(exported_file):
        print(f"Exported file not found: {exported_file}")
        print("Running export script...")
        try:
            subprocess.check_call(["python3", "../export-benchmarks.py"])
        except subprocess.CalledProcessError:
            print("Failed to export benchmarks")
            return False

    if not os.path.exists(exported_file):
        print(f"Exported file still not found after export: {exported_file}")
        return False

    with open(exported_file, "r") as f:
        content = f.read()

    lines = content.splitlines()
    i = 0
    found_function = None

    while i < len(lines):
        line = lines[i]
        func_def_pattern = re.compile(rf"^\s*(.*?)\s+({func_name})\s*\((.*?)\)\s*\{{\s*$")
        match = func_def_pattern.match(line)
        if match:
            return_type = match.group(1).strip()
            params = match.group(3).strip()
            comments = []
            j = i - 1
            while j >= 0:
                prev_line = lines[j]
                if prev_line.strip().startswith("//"):
                    comments.insert(0, prev_line)
                    j -= 1
                elif prev_line.strip() == "":
                    j -= 1
                else:
                    break
            func_body_lines = [line]
            brace_level = line.count("{") - line.count("}")
            i += 1
            while i < len(lines) and brace_level > 0:
                func_line = lines[i]
                func_body_lines.append(func_line)
                brace_level += func_line.count("{")
                brace_level -= func_line.count("}")
                i += 1
            func_body = "\n".join(func_body_lines)
            comments_str = "\n".join(comments)
            found_function = {
                "comments": comments_str,
                "return_type": return_type,
                "func_name": func_name,
                "params": params,
                "func_body": func_body,
            }
            break
        else:
            i += 1

    if not found_function:
        print(f"Function {func_name} not found in {exported_file}")
        return False

    func_body_lines = found_function["func_body"].split("\n")
    return_type = found_function["return_type"]
    params = found_function["params"]

    func_signature_line = f"__attribute__((noinline))\n{return_type} example({params}) {{"
    func_body_lines[0] = func_signature_line
    func_code = found_function["comments"] + "\n" + "\n".join(func_body_lines)

    includes = "#include <math.h>\n#include <stdint.h>\n#define TRUE 1\n#define FALSE 0\n"
    example_c_content = includes + "\n" + func_code

    output_file = os.path.join(tmp_dir, f"{prefix}example.c")
    with open(output_file, "w") as f:
        f.write(example_c_content)

    print(f"Extracted and renamed function {func_name} -> example to {output_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run cost-model ablation study.")
    parser.add_argument("--prefix", type=str, default="fptaylor-extra-ex11-", help="Prefix for intermediate files")
    parser.add_argument("--plot-only", action="store_true", help="Plot results from existing data")
    parser.add_argument("--output-format", type=str, default="png", help="Output format for plots")
    parser.add_argument("--num-parallel", type=int, default=16, help="Number of parallel processes")
    parser.add_argument(
        "--ablation-type", type=str, choices=["cost-model"], default="cost-model", help="Type of ablation study"
    )
    parser.add_argument("--show-prediction", action="store_true", help="Show predicted results in a second subplot")

    args = parser.parse_args()

    original_prefix = args.prefix
    if not original_prefix.endswith("-"):
        original_prefix += "-"

    tmp_dir = "tmp"
    logs_dir = "logs"
    plots_dir = "plots"

    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    if args.plot_only:
        plot_ablation_results_cost_model(
            tmp_dir,
            plots_dir,
            original_prefix,
            args.output_format,
            show_prediction=args.show_prediction,
        )
        sys.exit(0)
    else:
        print("=== Running cost-model ablation study ===")

        if not extract_function_from_exported(original_prefix, tmp_dir):
            print("Failed to extract function from exported files")
            sys.exit(1)

        remove_cache_dir()
        FPOPTFLAGS_CUSTOM = FPOPTFLAGS_BASE_TEMPLATE.copy()
        prefix_custom = f"{original_prefix}abl-custom-"

        data_custom = build_with_benchmark(
            tmp_dir,
            logs_dir,
            plots_dir,
            original_prefix,
            prefix_custom,
            FPOPTFLAGS_CUSTOM,
            num_parallel=args.num_parallel,
        )

        remove_cache_dir()
        FPOPTFLAGS_TTI = remove_mllvm_flag(FPOPTFLAGS_CUSTOM, "--fpopt-cost-model-path=")
        prefix_tti = f"{original_prefix}abl-tti-"

        data_tti = build_with_benchmark(
            tmp_dir,
            logs_dir,
            plots_dir,
            original_prefix,
            prefix_tti,
            FPOPTFLAGS_TTI,
            num_parallel=args.num_parallel,
        )

        all_data = {
            "custom": data_custom,
            "tti": data_tti,
        }

        ablation_data_file = os.path.join(tmp_dir, f"{original_prefix}ablation-cost-model.pkl")
        with open(ablation_data_file, "wb") as f:
            pickle.dump(all_data, f)
        print(f"Ablation data saved to {ablation_data_file}")

        plot_ablation_results_cost_model(
            tmp_dir,
            plots_dir,
            original_prefix,
            args.output_format,
            show_prediction=args.show_prediction,
        )

        clean_tmp_except_pkl(tmp_dir)


if __name__ == "__main__":
    main()
