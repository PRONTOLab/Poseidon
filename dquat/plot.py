#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from matplotlib import rcParams

sns.set_theme(style="whitegrid")
rcParams["axes.labelsize"] = 24
rcParams["xtick.labelsize"] = 20
rcParams["ytick.labelsize"] = 20
rcParams["legend.fontsize"] = 20

pkl_files = {"PT + AR": "both.pkl", "PT": "pt.pkl", "AR": "rewrites.pkl"}


def load_data(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    budgets = np.array(data["budgets"])
    runtimes = np.array(data["runtimes"])
    errors = np.array(data["errors"])
    worst_key = "max_errors" if "max_errors" in data else ("worst_errors" if "worst_errors" in data else None)
    if worst_key is not None:
        worst_errors = np.array(data[worst_key])
    else:
        worst_errors = np.full_like(errors, np.nan, dtype=float)
    original_runtime = data.get("original_runtime", np.nan)
    original_error = data.get("original_error", np.nan)
    original_worst_error = data.get("original_max_error", data.get("original_worst_error", np.nan))
    return budgets, runtimes, errors, worst_errors, original_runtime, original_error, original_worst_error


def process_data(budgets, runtimes, errors, worst_errors=None, filter_budgets=None):
    mask = np.ones(len(runtimes), dtype=bool)
    if filter_budgets is not None:
        mask = mask & np.array([b not in filter_budgets for b in budgets])
    budgets = budgets[mask]
    runtimes = runtimes[mask]
    errors = errors[mask]
    if worst_errors is not None:
        worst_errors = worst_errors[mask]
    unique = {}
    iterator = zip(
        budgets,
        runtimes,
        errors,
        worst_errors if worst_errors is not None else [None] * len(budgets),
    )
    for b, rt, err, worst in iterator:
        if b in unique:
            if err < unique[b][1]:
                unique[b] = (rt, err, worst)
        else:
            unique[b] = (rt, err, worst)
    proc_budgets = np.array(list(unique.keys()))
    proc_runtimes = np.array([v[0] for v in unique.values()])
    proc_errors = np.array([v[1] for v in unique.values()])
    proc_worst = np.array([v[2] for v in unique.values()]) if worst_errors is not None else None
    return proc_budgets, proc_runtimes, proc_errors, proc_worst


def compute_pareto_front(budgets, runtimes, errors, orig_runtime, orig_error):
    points = np.column_stack((runtimes, errors, budgets))
    orig_point = np.array([orig_runtime, orig_error, -1])
    if not np.any(np.all(np.isclose(points[:, :2], orig_point[:2], atol=1e-12), axis=1)):
        points = np.vstack((points, orig_point))
    sorted_points = points[np.argsort(points[:, 0])]
    pareto = []
    best_error = np.inf
    for p in sorted_points:
        if p[1] < best_error:
            pareto.append(p)
            best_error = p[1]
    return np.array(pareto)


def main():
    for label, fname in pkl_files.items():
        if not os.path.exists(fname):
            print(f"Pickle file '{fname}' not found!")
            return

    budgets_both, runtimes_both, errors_both, worst_both, orig_runtime, orig_error, orig_worst_error = load_data(
        pkl_files["PT + AR"]
    )
    proc_budgets_both, proc_runtimes_both, proc_errors_both, proc_worst_both = process_data(
        budgets_both, runtimes_both, errors_both, worst_both
    )
    b0_idx = np.where(proc_budgets_both == 0)[0]
    if b0_idx.size > 0:
        orig_runtime = float(proc_runtimes_both[b0_idx[0]])
        orig_error = float(proc_errors_both[b0_idx[0]])
        if proc_worst_both is not None and len(proc_worst_both) > int(b0_idx[0]):
            orig_worst_error = float(proc_worst_both[b0_idx[0]])
    orig_point = np.array([orig_runtime, orig_error])
    pareto_fronts = {}
    pareto_fronts["PT + AR"] = compute_pareto_front(
        proc_budgets_both, proc_runtimes_both, proc_errors_both, orig_runtime, orig_error
    )

    both_dict = {
        b: (rt, err, worst)
        for b, rt, err, worst in zip(proc_budgets_both, proc_runtimes_both, proc_errors_both, proc_worst_both)
    }
    both_budgets_set = set(proc_budgets_both.tolist())
    worst_lookup = {}
    worst_lookup["PT + AR"] = {b: w for b, w in zip(proc_budgets_both, proc_worst_both)}

    for label in ["PT", "AR"]:
        budgets, runtimes, errors, worst, _, _, _ = load_data(pkl_files[label])
        proc_budgets, proc_runtimes, proc_errors, proc_worst = process_data(budgets, runtimes, errors, worst)
        updated_runtimes = []
        updated_errors = []
        updated_worst = []
        for b, rt, err in zip(proc_budgets, proc_runtimes, proc_errors):
            if b in both_dict:
                print(f"Updating budget {b} from {rt:.4f}, {err:.16e} to {both_dict[b][0]:.4f}, {both_dict[b][1]:.16e}")
                new_rt, new_err, new_worst = both_dict[b]
                updated_runtimes.append(new_rt)
                updated_errors.append(new_err)
                updated_worst.append(new_worst)
            else:
                updated_runtimes.append(rt)
                updated_errors.append(err)
                idx = np.where(proc_budgets == b)[0]
                if len(idx) > 0 and proc_worst is not None:
                    updated_worst.append(proc_worst[idx[0]])
                else:
                    updated_worst.append(np.nan)
        proc_runtimes = np.array(updated_runtimes)
        proc_errors = np.array(updated_errors)
        proc_worst = np.array(updated_worst)
        pareto_fronts[label] = compute_pareto_front(proc_budgets, proc_runtimes, proc_errors, orig_runtime, orig_error)
        worst_lookup[label] = {b: w for b, w in zip(proc_budgets, proc_worst)}

    plt.figure(figsize=(10, 6))
    marker_style = "*"
    colors = {"PT + AR": "#2287E6", "PT": "#008001", "AR": "#FFBD59"}
    width = {"PT + AR": 5, "PT": 4.5, "AR": 4.5}
    linestyle = {"PT + AR": "solid", "PT": "dotted", "AR": "dashed"}
    marker_size = 100
    label_offsets = {"PT + AR": (6, 8), "PT": (6, -18), "AR": (-28, 8)}

    for label, front in pareto_fronts.items():
        front = front[np.argsort(front[:, 0])]
        plt.step(
            front[:, 0],
            front[:, 1],
            linestyle=linestyle[label],
            color=colors[label],
            label=label,
            where="post",
            linewidth=width[label],
            alpha=0.7,
            zorder=1,
        )
        print(f"Pareto front for {label}:")
        print(front)
        print("Worst-case errors per Pareto point:")
        for point in front:
            if np.isclose(point[2], -1):
                wc = orig_worst_error
                bstr = "Baseline"
            else:
                b = int(point[2])
                wc = worst_lookup.get(label, {}).get(b, np.nan)
                bstr = f"Budget {b}"
            if wc is None or np.isnan(wc):
                print(f"  {bstr}: N/A")
            else:
                print(f"  {bstr}: {wc:.16e}")
        for point in front:
            if np.isclose(point[2], -1):
                continue
            if label in ["PT", "AR"] and (point[2] in both_budgets_set):
                continue
            marker_color = colors[label]
            plt.scatter(point[0], point[1], marker=marker_style, color=marker_color, s=marker_size, zorder=1)

    plt.plot(
        orig_point[0], orig_point[1], marker="^", markersize=10, color="#FF6666", linestyle="None", label="Baseline"
    )
    print(f"Original Runtime: {orig_point[0]:.4f}, Original Error: {orig_point[1]:.16e}")
    plt.xlabel("Runtime (second)")
    plt.ylabel("Relative Error")
    plt.yscale("log")

    plt.legend(loc="lower left")

    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.savefig("dquat.png")
    plt.show()


if __name__ == "__main__":
    main()
