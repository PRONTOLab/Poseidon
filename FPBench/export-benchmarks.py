#!/usr/bin/env python3

import os
import subprocess
import sys
import glob
import argparse


def export_fpcore_to_c(fpcore_file, output_dir, racket_script, force_regen=False):
    filename = os.path.basename(fpcore_file)
    base_name = os.path.splitext(filename)[0]
    c_filename = f"{base_name}.fpcore.c"
    c_filepath = os.path.join(output_dir, c_filename)

    if not force_regen and os.path.exists(c_filepath):
        print(f"{c_filename} already exists. Skipping generation.")
        return c_filepath

    print(f"Generating {c_filename} using Racket script...")
    try:
        print("Running command: ", " ".join(["racket", racket_script, fpcore_file, c_filepath]))
        subprocess.check_call(["racket", racket_script, fpcore_file, c_filepath])
        print(f"Generated {c_filename} successfully.")
        return c_filepath
    except subprocess.CalledProcessError:
        print(f"Error running export.rkt on {filename}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Export FPCore benchmarks to C files.")
    parser.add_argument("--regen", action="store_true", help="Force regeneration of .c files")
    parser.add_argument("--source-dir", type=str, default="benchmarks", help="Source directory for .fpcore files")
    parser.add_argument("--output-dir", type=str, default="exported", help="Output directory for .c files")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(script_dir, args.source_dir)
    output_dir = os.path.join(script_dir, args.output_dir)
    racket_script = os.path.join(script_dir, "export.rkt")

    if not os.path.exists(racket_script):
        print(f"Error: Racket script not found at {racket_script}")
        sys.exit(1)

    fpcore_files = glob.glob(os.path.join(source_dir, "*.fpcore"))

    if not fpcore_files:
        print(f"No .fpcore files found in {source_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    exported_files = []
    for fpcore_file in fpcore_files:
        c_filepath = export_fpcore_to_c(fpcore_file, output_dir, racket_script, args.regen)
        if c_filepath:
            exported_files.append(c_filepath)

    print(f"\nExported {len(exported_files)} files to {output_dir}")


if __name__ == "__main__":
    main()
