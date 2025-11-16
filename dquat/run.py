#!/usr/bin/env python3

import os
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

NUM_PARALLEL = 32

HOME = "/root"

LLVM_PATH = os.path.join(HOME, "Poseidon/llvm-project/build/bin")
ENZYME_PATH = os.path.join(HOME, "Poseidon/Enzyme/build/Enzyme/ClangEnzyme-22.so")
PROFILER_PATH = os.path.join(HOME, "Poseidon/Enzyme/build/Enzyme")

CXX = os.path.join(LLVM_PATH, "clang++")

CXXFLAGS = [
    "-O3",
    "-I.",
    "-Wall",
    f"-I{os.path.join(HOME, 'include')}",
    f"-L{os.path.join(HOME, 'lib')}",
    "-I/usr/include/c++/13",
    "-I/usr/include/x86_64-linux-gnu/c++/13",
    "-L/usr/lib/gcc/x86_64-linux-gnu/13",
    "-fno-exceptions",
    f"-fpass-plugin={ENZYME_PATH}",
    "-Xclang",
    "-load",
    "-Xclang",
    ENZYME_PATH,
    "-ffast-math",
    "-march=native",
]

LINKFLAGS = [
    "-lmpfr",
    "-lm",
]

FPOPTFLAGS_BASE = [
    "-mllvm",
    "--fpprofile-use=./fpprofile",
    "-mllvm",
    "--fpopt-enable-herbie=1",
    "-mllvm",
    "--fpopt-enable-solver",
    "-mllvm",
    "--fpopt-enable-pt",
    "-mllvm",
    "--fpopt-cost-dom-thres=0.0",
    "-mllvm",
    "--fpopt-acc-dom-thres=0.0",
    "-mllvm",
    "--fpopt-early-prune",
    "-mllvm",
    "--fpopt-comp-cost-budget={budget}",
    "-mllvm",
    "--fpopt-cache-path=cache",
    "-mllvm",
    "--fpopt-num-samples=1024",
    "-mllvm",
    f"--fpopt-cost-model-path={os.path.join(HOME, 'Poseidon/cost-model/cm.csv')}",
    "-mllvm",
    "--fpopt-strict-mode",
]


LOG_DIR = "logs"
OUTPUT_DIR = "tmp"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_command(command, log_file):
    try:
        with open(log_file, "w") as f:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
            f.write(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        with open(log_file, "a") as f:
            f.write("\nError:\n")
            f.write(e.stdout)
        return False


def load_budgets(file_path):
    try:
        with open(file_path, "r") as f:
            line = f.readline()
            budgets = [int(budget.strip()) for budget in line.split(",") if budget.strip()]
        return budgets
    except FileNotFoundError:
        print(f"Budget file {file_path} not found.")
        return []
    except ValueError as e:
        print(f"Error parsing budgets from {file_path}: {e}")
        return []


def compile_fpopt_with_profile(budget, profile_path="fpprofile"):
    fpopt_flags = FPOPTFLAGS_BASE.copy()
    fpopt_flags = [flag if "{budget}" not in flag else flag.format(budget=budget) for flag in fpopt_flags]

    fpopt_flags = [
        flag if flag != "--fpprofile-use=./fpprofile" else f"--fpprofile-use={profile_path}" for flag in fpopt_flags
    ]

    output_name = f"dquat-fpopt-{budget}"
    if profile_path != "fpprofile":
        profile_suffix = os.path.basename(profile_path).replace("fpprofile_", "")
        output_name = f"dquat-fpopt-{budget}-prof{profile_suffix}"

    command = (
        [CXX] + CXXFLAGS + fpopt_flags + ["dquat.cpp", "-o", os.path.join(OUTPUT_DIR, f"{output_name}.exe")] + LINKFLAGS
    )

    log_file = os.path.join(LOG_DIR, f"compile_{output_name}.log")

    success = run_command(command, log_file)

    return (budget, success, log_file, profile_path)


def main():
    parser = argparse.ArgumentParser(description="Compile optimized executables with fpprofile")
    parser.add_argument(
        "--profile-size", type=int, default=100, help="Number of samples for profile generation (default: 1000000)"
    )

    num_workers = NUM_PARALLEL

    BUDGET_PATH = "cache/budgets.txt"
    budgets = load_budgets(BUDGET_PATH)

    if not budgets:
        print("No budgets found. Exiting.")
        return

    print(f"\nLoaded {len(budgets)} budgets from {BUDGET_PATH}")

    compiled = []
    failed = []

    print(f"\nStarting parallel compilation with {len(budgets)} budgets using {num_workers} workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_budget = {executor.submit(compile_fpopt_with_profile, budget): budget for budget in budgets}

        for future in tqdm(as_completed(future_to_budget), total=len(future_to_budget), desc="Compiling"):
            budget = future_to_budget[future]
            try:
                bud, success, log, prof = future.result()
                if success:
                    compiled.append(budget)
                else:
                    failed.append(budget)
            except Exception as exc:
                print(f"Budget {budget} generated an exception: {exc}")
                failed.append(budget)

    print("\nCompilation Summary:")
    print(f"Total Budgets: {len(budgets)}")
    print(f"Successfully Compiled: {len(compiled)}")
    print(f"Failed Compilations: {len(failed)}")

    if failed:
        print("\nFailed Budgets:")
        for budget in failed:
            print(f"  - {budget} (See logs in {LOG_DIR})")


if __name__ == "__main__":
    main()
