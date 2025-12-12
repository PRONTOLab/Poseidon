#!/usr/bin/env python3
import glob
import re
import subprocess
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from decimal import Decimal, getcontext
import matplotlib.lines as mlines

getcontext().prec = 600

def modify_real_type(real_type):
    with open("lulesh.h", "r") as f:
        content = f.read()
    new_content = re.sub(
        r"typedef\s+(real4|real8|real10)\s+Real_t\s*;",
        f"typedef {real_type}   Real_t ;",
        content
    )
    with open("lulesh.h", "w") as f:
        f.write(new_content)

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running: {cmd}")
        print(result.stderr)
    return result.returncode == 0

def generate_real_file(real_type, output_file):
    print(f"Generating {output_file}...")
    modify_real_type(real_type)
    if not run_command("make clean"):
        return False
    if not run_command("make ser-single-forward.exe"):
        return False
    os.makedirs("ablation", exist_ok=True)
    return run_command(f'script -c "./ser-single-forward.exe -s 50 -e" {output_file}')

def generate_fpopt_file(output_file):
    print(f"Generating {output_file}...")
    modify_real_type("real8")
    if not run_command("make clean"):
        return False
    if not run_command("make ser-single-forward-fpopt.exe"):
        return False
    os.makedirs("ablation", exist_ok=True)
    return run_command(f'script -c "./ser-single-forward-fpopt.exe -s 50 -e" {output_file}')

if __name__ == "__main__":
    generate_real_file("real4", "ablation/real4.txt")
    generate_real_file("real8", "ablation/real8.txt")
    generate_real_file("real10", "ablation/real10.txt")
    generate_fpopt_file("ablation/fpopt.txt")

    modify_real_type("real8")

    sns.set_theme(style="whitegrid")
    rcParams["font.size"] = 20
    rcParams["axes.titlesize"] = 24
    rcParams["axes.labelsize"] = 20
    rcParams["xtick.labelsize"] = 18
    rcParams["ytick.labelsize"] = 18
    rcParams["legend.fontsize"] = 18

    blue = "#2287E6"
    yellow = "#FFBD59"
    red = "#FF6666"

    files = glob.glob("ablation/*.txt")
    data = {}
    for filepath in files:
        filename = os.path.basename(filepath)
        with open(filepath, "r") as f:
            content = f.read()
        m_cycle = re.search(r"Cycle\s+#1662:\s*([0-9eE+.\-]+)", content)
        m_time = re.search(r"Elapsed time\s*=\s*([0-9eE+.\-]+)", content)
        if m_cycle and m_time:
            value = Decimal(m_cycle.group(1))
            elapsed = float(m_time.group(1))
        else:
            continue
        if filename in ["real4.txt", "real8.txt", "real10.txt", "fpopt.txt"]:
            if filename == "real4.txt":
                sig = 24
            elif filename == "real8.txt":
                sig = 53
            elif filename == "real10.txt":
                sig = 64
            elif filename == "fpopt.txt":
                sig = 53
        elif filename.startswith("mpfr"):
            m = re.search(r"mpfr(\d+)", filename)
            sig = int(m.group(1)) if m else None
        else:
            sig = None
        data[filename] = {"value": value, "elapsed": elapsed, "sig": sig}

    if "mpfr512.txt" not in data:
        raise RuntimeError("mpfr512.txt not found. It is required as the reference for relative error.")
    ref_value = data["mpfr512.txt"]["value"]

    if "real8.txt" not in data:
        raise RuntimeError("real8.txt not found. It is required for speedup computation.")
    ref_time_speedup = data["real8.txt"]["elapsed"]

    plt.figure(figsize=(10, 6))
    legend_handles = {}

    for fname, info in data.items():
        if info["sig"] is None:
            continue
        if fname in ["mpfr512.txt"]:
            continue
        rel_error = abs(info["value"] - ref_value) / abs(ref_value)
        slowdown = info["elapsed"] / ref_time_speedup
        x = info["sig"]
        y = float(rel_error)
        if fname.startswith("mpfr"):
            color = yellow
            marker = "*"
            category = "MPFR"
            annotation_text = "$> 650\\times$" if fname == "mpfr65.txt" else None
        elif fname == "fpopt.txt":
            color = blue
            marker = "*"
            category = "Optimized"
            annotation_text = f"$ {slowdown:.2f}\\times$"
        else:
            color = red
            marker = "^"
            category = "Original"
            annotation_text = f"$ {slowdown:.2f}\\times$"
        label = None
        if category not in legend_handles:
            label = category
        scatter = plt.scatter(x, y, color=color, marker=marker, s=150, zorder=5, label=label)
        if label is not None:
            legend_handles[category] = scatter
        if annotation_text is not None:
            plt.annotate(annotation_text, xy=(x, y), xytext=(5, 5), textcoords="offset points", fontsize=16)

    for fname in ["real8.txt", "real10.txt", "fpopt.txt"]:
        if fname not in data:
            continue
        info = data[fname]
        rel_error = abs(info["value"] - ref_value) / abs(ref_value)
        if fname == "fpopt.txt":
            line_color = blue
        else:
            line_color = red
        plt.axhline(y=float(rel_error), color=line_color, linestyle="--", linewidth=1)

    plt.xlabel(r"# Significands")
    plt.ylabel("Relative Error")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    category_order = ["Original", "Optimized", "MPFR"]
    display_label_map = {
        "Original": "Native FP Baseline",
        "Optimized": "Native FP Optimized",
        "MPFR": "MPFR Baseline",
    }
    handles = [legend_handles[cat] for cat in category_order if cat in legend_handles]
    labels = [display_label_map[cat] for cat in category_order if cat in legend_handles]

    dummy_handle = mlines.Line2D([], [], color="none", marker="", linestyle="None", markersize=0, label="Anno. = Slowdown")
    handles.append(dummy_handle)
    labels.append("Anno. = Slowdown")
    plt.legend(handles=handles, labels=labels)

    plt.tight_layout()
    plt.savefig("lulesh.png")
    plt.show()
