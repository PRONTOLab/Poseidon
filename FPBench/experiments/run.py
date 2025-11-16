import os
import subprocess
import sys
import shutil
import re
import argparse
import math
import random
from statistics import mean
import pickle

from tqdm import trange
from concurrent.futures import ProcessPoolExecutor, as_completed

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

FPOPTFLAGS_BASE = [
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
EXE = ["example.exe", "example-logged.exe", "example-fpopt.exe"]
NUM_RUNS = 10
DRIVER_NUM_SAMPLES = 100000
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


def clean(tmp_dir, logs_dir):
    print("=== Cleaning up generated files ===")
    directories = [tmp_dir, logs_dir]
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"Removed directory: {directory}")


def clean_tmp_except_pkl(tmp_dir):
    for entry in os.listdir(tmp_dir):
        full_path = os.path.join(tmp_dir, entry)
        if os.path.isfile(full_path) and not full_path.endswith(".pkl"):
            os.remove(full_path)
            print(f"Removed file: {full_path}")
        elif os.path.isdir(full_path):
            shutil.rmtree(full_path)
            print(f"Removed directory: {full_path}")


def generate_example_cpp(tmp_dir, prefix):
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fpopt-original-driver-generator.py")
    print(f"=== Running {script} ===")
    src_prefixed = os.path.join(tmp_dir, f"{prefix}{SRC}")
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

    if "rosa-ex36" in prefix:
        cmd.insert(-2, "-fno-unroll-loops")

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


def generate_golden_values(tmp_dir, prefix):
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fpopt-golden-driver-generator.py")
    src_prefixed = os.path.join(tmp_dir, f"{prefix}{SRC}")
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


def build_all(tmp_dir, prefix):
    generate_example_cpp(tmp_dir, prefix)
    compile_example_exe(tmp_dir, prefix)
    compile_example_logged_exe(tmp_dir, prefix)
    generate_fpprofile(tmp_dir, prefix)
    fpoptflags = FPOPTFLAGS_BASE.copy()
    fpoptflags.append("-mllvm")
    fpoptflags.append(f"--fpprofile-use=tmp/{prefix}fpprofile")
    compile_example_fpopt_exe(tmp_dir, prefix, fpoptflags, output="example-fpopt.exe")
    print("=== Initial build process completed successfully ===")


def process_cost(args):
    cost, tmp_dir, prefix = args

    print(f"\n=== Processing computation cost budget: {cost} ===")
    fpoptflags = []
    for flag in FPOPTFLAGS_BASE:
        if flag.startswith("--fpopt-comp-cost-budget="):
            fpoptflags.append(f"--fpopt-comp-cost-budget={cost}")
        else:
            fpoptflags.append(flag)
    fpoptflags.append("-mllvm")
    fpoptflags.append(f"--fpprofile-use=tmp/{prefix}fpprofile")

    output_binary = f"example-fpopt-{cost}.exe"

    compile_example_fpopt_exe(tmp_dir, prefix, fpoptflags, output=output_binary, verbose=False)

    generate_values(tmp_dir, prefix, output_binary)

    return cost, output_binary


def benchmark(tmp_dir, prefix, num_parallel=1):
    costs = parse_critical_comp_costs(tmp_dir, prefix)

    original_avg_runtime = measure_runtime(tmp_dir, prefix, "example.exe", NUM_RUNS)
    original_runtime = original_avg_runtime

    if original_runtime is None:
        print("Original binary timed out. Proceeding as if it doesn't exist.")
        return

    generate_example_values(tmp_dir, prefix)

    generate_golden_values(tmp_dir, prefix)

    golden_values_file = get_values_file_path(tmp_dir, prefix, "golden.exe")
    example_binary = "example.exe"
    rel_errs_example = get_avg_rel_error(tmp_dir, prefix, golden_values_file, [example_binary])
    rel_err_example = rel_errs_example[example_binary]
    print(f"Average Rel Error for {prefix}example.exe: {rel_err_example}")

    data_tuples = []

    args_list = [(cost, tmp_dir, prefix) for cost in costs]

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
    data_file = os.path.join(tmp_dir, f"{prefix}benchmark_data.pkl")
    with open(data_file, "wb") as f:
        pickle.dump(data, f)
    print(f"Benchmark data saved to {data_file}")


def analyze_all_data(tmp_dir, thresholds=None):
    data_list = []

    for filename in os.listdir(tmp_dir):
        if filename.endswith("benchmark_data.pkl"):
            data_file = os.path.join(tmp_dir, filename)
            with open(data_file, "rb") as f:
                data = pickle.load(f)
            prefix = filename[: -len("benchmark_data.pkl")]
            data_list.append((prefix, data))

    print("Number of tested FPBench functions: ", len(data_list))
    if not data_list:
        print("No benchmark data files found in the tmp directory.")
        return

    if thresholds is None:
        thresholds = [
            0,
            1e-15,
            1e-14,
            1e-13,
            1e-12,
            1e-11,
            1e-10,
            1e-9,
            1e-8,
            1e-7,
            1e-6,
            1e-5,
            5e-5,
            1e-4,
            1e-3,
            1e-2,
            1e-1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.9,
            1,
        ]

    min_runtime_ratios = {threshold: {} for threshold in thresholds}

    for prefix, data in data_list:
        runtimes = data["runtimes"]
        errors = data["errors"]
        original_runtime = data["original_runtime"]

        for threshold in thresholds:
            min_ratio = None
            for err, runtime in zip(errors, runtimes):
                if err is not None and runtime is not None and err <= threshold:
                    runtime_ratio = runtime / original_runtime
                    if min_ratio is None or runtime_ratio < min_ratio:
                        min_ratio = runtime_ratio
            if min_ratio is not None:
                min_runtime_ratios[threshold][prefix] = min_ratio

    overall_runtime_improvements = {}
    for threshold in thresholds:
        ratios = min_runtime_ratios[threshold].values()
        if ratios:
            log_sum = sum(math.log(min(1, ratio)) for ratio in ratios)
            geo_mean_ratio = math.exp(log_sum / len(ratios))
            percentage_improvement = (1 - geo_mean_ratio) * 100
            overall_runtime_improvements[threshold] = percentage_improvement
        else:
            overall_runtime_improvements[threshold] = None

    max_speedups = {}
    max_speedup_prefixes = {}
    for threshold in thresholds:
        if min_runtime_ratios[threshold]:
            best_prefix = min(min_runtime_ratios[threshold], key=min_runtime_ratios[threshold].get)
            best_ratio = min_runtime_ratios[threshold][best_prefix]
            max_speedup = 1 / best_ratio if best_ratio > 0 else float("inf")
            max_speedups[threshold] = max_speedup
            max_speedup_prefixes[threshold] = best_prefix
        else:
            max_speedups[threshold] = None
            max_speedup_prefixes[threshold] = None

    print("\nMaximum speedup on a single benchmark for each threshold:")
    for threshold in thresholds:
        prefix = max_speedup_prefixes[threshold]
        if prefix is not None:
            print(f"Allowed relative error ≤ {threshold}: {max_speedups[threshold]:.2f}x speedup ({prefix})")
        else:
            print(f"Allowed relative error ≤ {threshold}: No data")

    max_accuracy_improvement_ratio = 0.0
    max_accuracy_improvement_prefix = None
    better_accuracy_count = 0

    for prefix, data in data_list:
        orig_err = data["original_error"]
        valid_optimized_errors = [err for err in data["errors"] if err is not None and err > 0]
        if valid_optimized_errors and orig_err is not None:
            best_optimized_error = min(valid_optimized_errors)
            if best_optimized_error < orig_err:
                better_accuracy_count += 1
                improvement_ratio = orig_err / best_optimized_error
                if improvement_ratio > max_accuracy_improvement_ratio:
                    max_accuracy_improvement_ratio = improvement_ratio
                    max_accuracy_improvement_prefix = prefix

    if max_accuracy_improvement_prefix is not None:
        print(
            f"\nMaximum accuracy improvement ratio: {max_accuracy_improvement_ratio:.2f}x (in benchmark: {max_accuracy_improvement_prefix})"
        )
    else:
        print("\nNo accuracy improvements found.")

    print(f"\nNumber of benchmarks where we can get better accuracy: {better_accuracy_count}")

    original_errors = [
        data["original_error"]
        for _, data in data_list
        if data["original_error"] is not None and data["original_error"] > 0
    ]

    if original_errors:
        log_sum = sum(math.log(err) for err in original_errors)
        geomean_error = math.exp(log_sum / len(original_errors))
        print("Geometric mean of original relative errors:", geomean_error)
    else:
        print("No valid original errors found.")

    reduction_ratios = []
    for prefix, data in data_list:
        orig_err = data["original_error"]
        valid_optimized_errors = [err for err in data["errors"] if err is not None and err > 0]
        if valid_optimized_errors and orig_err is not None and orig_err > 0:
            best_optimized_error = min(valid_optimized_errors)
            if best_optimized_error < orig_err:
                reduction_ratio = best_optimized_error / orig_err
                reduction_ratios.append(reduction_ratio)

    if reduction_ratios:
        log_sum = sum(math.log(ratio) for ratio in reduction_ratios)
        geomean_reduction = math.exp(log_sum / len(reduction_ratios))
        improvement_factor = 1 / geomean_reduction
        print("\nGeometric mean of error reduction:", geomean_reduction)
        print("Average improvement factor in error:", improvement_factor, "x")
    else:
        print("\nNo error improvements found.")


def remove_cache_dir():
    cache_dir = "cache"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print("=== Removed existing cache directory ===")


def main():
    parser = argparse.ArgumentParser(description="Run the example C code with prefix handling.")
    parser.add_argument("--prefix", type=str, help="Prefix for intermediate files (e.g., rosa-ex23-)")
    parser.add_argument("--analytics", action="store_true", help="Run analytics on saved data")
    parser.add_argument(
        "--num-parallel", type=int, default=16, help="Number of parallel processes to use (default: 16)"
    )
    args = parser.parse_args()

    if not args.analytics and args.prefix is None:
        parser.error("--prefix is required for all operations except --analytics")

    prefix = args.prefix if args.prefix else ""
    if prefix and not prefix.endswith("-"):
        prefix += "-"

    tmp_dir = "tmp"
    logs_dir = "logs"

    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    remove_cache_dir()

    if args.analytics:
        analyze_all_data(tmp_dir)
    else:
        build_all(tmp_dir, prefix)
        benchmark(tmp_dir, prefix, num_parallel=args.num_parallel)
        clean_tmp_except_pkl(tmp_dir)


if __name__ == "__main__":
    main()
