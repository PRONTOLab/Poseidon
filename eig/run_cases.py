#!/usr/bin/env python3

import argparse
import os
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


def run_command(cmd: str, description: str):
    print(f"\n{'='*50}")
    print(f"{description}")
    print(f"Command: {cmd}")
    print("=" * 50)

    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Warning: Command exited with code {result.returncode}")
    return result.returncode


def run_case(case_name: str):
    cache_dir = f"cache-{case_name}"

    print(f"\n{'#'*60}")
    print(f"# Running case: {case_name}")
    print(f"{'#'*60}")

    print(f"\nSetting up cache directory from {cache_dir}...")
    if os.path.exists("cache"):
        shutil.rmtree("cache")
    if os.path.exists(cache_dir):
        shutil.copytree(cache_dir, "cache")
    else:
        print(f"Warning: {cache_dir} not found, skipping cache setup")

    print("\nCleaning up tmp, logs, outputs directories...")
    for d in ["tmp", "logs", "outputs"]:
        if os.path.exists(d):
            shutil.rmtree(d)

    run_command("make clean", "Cleaning build")
    run_command(f"make CASE={case_name}", "Building")

    run_command("python3 run.py", "Running run.py")
    run_command("python3 benchmark.py", "Running benchmark.py")

    stat_file = f"{case_name}.txt"
    run_command(f'script -c "python3 benchmark.py --analyze-only" {stat_file}', f"Saving results to {stat_file}")

    print(f"\nCase {case_name} completed!")


def main():
    parser = argparse.ArgumentParser(description="Run benchmark cases for eig")
    parser.add_argument(
        "--case",
        choices=["biased", "equal"],
        action="append",
        help="Case to run (can specify multiple times, default: both)",
    )
    args = parser.parse_args()

    if args.case:
        cases_to_run = args.case
    else:
        cases_to_run = ["biased", "equal"]

    for case_name in cases_to_run:
        run_case(case_name)

    print(f"\n{'='*60}")
    print("All cases completed!")
    print(f"Results saved to: {', '.join(c + '.txt' for c in cases_to_run)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
