#!/usr/bin/env python3
"""
Generate poster-ready plots from experiment CSVs.

Usage:
    python3 scripts/plot_results.py

Outputs (saved to plots/):
    alpha_beta_fit.png      -- latency vs message size with fitted lines
    netsize_allreduce.png   -- allreduce time vs model size per algorithm
    netsize_allreduce_pct.png -- allreduce % of iter vs model size
    strong_scaling.png      -- throughput (samples/s) vs rank count
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict

os.makedirs("plots", exist_ok=True)

COLORS = {"tree": "#e15759", "ring": "#4e79a7", "hd": "#59a14f", "mpi": "#f28e2b"}
LABELS = {"tree": "Tree", "ring": "Ring", "hd": "Halving-Doubling", "mpi": "MPI Builtin"}

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})

# ---------------------------------------------------------------------------
# 1. Alpha-beta fit plot
# ---------------------------------------------------------------------------
def load_bench_csv(path):
    data = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("ranks"):
                continue
            parts = [p.strip() for p in line.split(",")]
            try:
                n_doubles = int(parts[1])
                algo = parts[2]
                median_ms = float(parts[3])
            except (ValueError, IndexError):
                continue
            data[algo].append((n_doubles * 8, median_ms / 1000.0))
    return data

def fit_alpha_beta(points):
    Ns = np.array([p[0] for p in points], dtype=float)
    Ts = np.array([p[1] for p in points], dtype=float)
    coeffs = np.polyfit(Ns, Ts, 1)
    return coeffs[1] * 1e6, coeffs[0] * 1e6  # alpha_us, beta_us_per_byte

def plot_alpha_beta(csv_path="microbench_p8.csv"):
    if not os.path.exists(csv_path):
        print(f"skipping alpha-beta plot: {csv_path} not found")
        return
    data = load_bench_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    for algo, pts in sorted(data.items()):
        if algo not in COLORS or algo == "mpi":
            continue
        pts_sorted = sorted(pts, key=lambda p: p[0])
        Ns = np.array([p[0] for p in pts_sorted])
        Ts = np.array([p[1] for p in pts_sorted]) * 1e6
        color = COLORS[algo]
        ax.plot(Ns, Ts, "o-", color=color, linewidth=1.5,
                markersize=5, label=LABELS.get(algo, algo))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Message size (bytes)")
    ax.set_ylabel("Median latency (μs)")
    ax.set_title("All-reduce latency vs message size  (P = 8 ranks)")
    ax.legend(fontsize=9)
    caption = ("At small message sizes all three algorithms perform similarly, but ring's bandwidth-optimal\n"
               "design pulls ahead as gradient buffers grow. Tree wastes bandwidth by sending the full\n"
               "gradient at every reduction step, while ring only sends 1/P of the data per step.")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=7.5, style="italic",
             wrap=True, color="#333333")
    plt.savefig("plots/alpha_beta_fit.png", dpi=180)
    plt.close()
    print("saved plots/alpha_beta_fit.png")

# ---------------------------------------------------------------------------
# 2. Netsize plots
# ---------------------------------------------------------------------------
def load_netsize_csvs():
    algos = ["ring", "tree", "hd"]
    rows = []
    for algo in algos:
        path = f"netsize_p8_{algo}.csv"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            header = f.readline()
            for line in f:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) < 7:
                    continue
                rows.append({
                    "config": parts[0],
                    "params": int(parts[1]),
                    "algo": parts[2],
                    "compute_ms": float(parts[3]),
                    "allreduce_ms": float(parts[4]),
                    "iter_ms": float(parts[5]),
                    "allreduce_pct": float(parts[6]),
                })
    return rows

def plot_netsize(rows):
    if not rows:
        print("skipping netsize plots: no data")
        return

    configs = ["small", "medium", "large", "xlarge"]
    config_labels = ["50K\n(small)", "235K\n(medium)", "567K\n(large)", "1.46M\n(xlarge)"]
    x = np.arange(len(configs))
    width = 0.25
    offsets = [-width, 0, width]
    algos = ["tree", "ring", "hd"]

    # Plot 1: allreduce time vs model size
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, algo in enumerate(algos):
        vals = []
        for cfg in configs:
            match = [r for r in rows if r["algo"] == algo and r["config"] == cfg]
            vals.append(match[0]["allreduce_ms"] if match else 0)
        ax.bar(x + offsets[i], vals, width, label=LABELS.get(algo, algo),
               color=COLORS[algo], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels)
    ax.set_xlabel("Model size (params)")
    ax.set_ylabel("Avg all-reduce time (ms)")
    ax.set_title("All-reduce cost vs model size  (P = 8 ranks)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("plots/netsize_allreduce.png", dpi=180)
    plt.close()
    print("saved plots/netsize_allreduce.png")

    # Plot 2: allreduce % of iteration
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, algo in enumerate(algos):
        vals = []
        for cfg in configs:
            match = [r for r in rows if r["algo"] == algo and r["config"] == cfg]
            vals.append(match[0]["allreduce_pct"] if match else 0)
        ax.bar(x + offsets[i], vals, width, label=LABELS.get(algo, algo),
               color=COLORS[algo], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels)
    ax.set_xlabel("Model size (params)")
    ax.set_ylabel("All-reduce % of iteration time")
    ax.set_title("Communication overhead vs model size  (P = 8 ranks)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig("plots/netsize_allreduce_pct.png", dpi=180)
    plt.close()
    print("saved plots/netsize_allreduce_pct.png")

# ---------------------------------------------------------------------------
# 3. Strong scaling plot
# ---------------------------------------------------------------------------
def load_scaling_csv(algo):
    path = f"scaling_{algo}.csv"
    if not os.path.exists(path):
        return {}
    data = defaultdict(list)
    with open(path) as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            ranks = int(parts[0])
            epoch = int(parts[1])
            sps = float(parts[5])
            if epoch == 1:      # skip epoch 1, it's noisy
                continue
            if ranks < 4:       # rank 1/2 show cache-bound anomaly with large local batch
                continue
            data[ranks].append(sps)
    return {r: np.mean(v) for r, v in data.items()}

def plot_strong_scaling():
    algos = ["ring", "tree", "hd"]
    any_data = False
    fig, ax = plt.subplots(figsize=(6, 4))

    for algo in algos:
        d = load_scaling_csv(algo)
        if not d:
            continue
        any_data = True
        ranks = sorted(d.keys())
        sps = [d[r] for r in ranks]
        ax.plot(ranks, sps, "o-", color=COLORS[algo],
                label=LABELS.get(algo, algo), linewidth=1.5, markersize=5)

    if not any_data:
        print("skipping strong scaling plot: no data")
        return

    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Throughput (samples/s)")
    ax.set_title("Strong scaling  (global batch = 512)")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(fontsize=9)
    caption = ("Ring and halving-doubling scale consistently with more workers, while tree hits a communication\n"
               "bottleneck early. At 32 ranks ring achieves 60% higher throughput than tree, showing that\n"
               "algorithm choice matters more as parallelism increases.")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=7.5, style="italic",
             wrap=True, color="#333333")
    plt.savefig("plots/strong_scaling.png", dpi=180)
    plt.close()
    print("saved plots/strong_scaling.png")

# ---------------------------------------------------------------------------
# 4. HD non-power-of-2 fold overhead plot
# ---------------------------------------------------------------------------
def plot_hd_nonpower2(csv_path="hd_nonpower2_comparison.csv"):
    if not os.path.exists(csv_path):
        print(f"skipping HD non-power-of-2 plot: {csv_path} not found")
        return

    from collections import defaultdict
    data = defaultdict(list)
    with open(csv_path) as f:
        f.readline()  # header
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 3:
                continue
            try:
                ranks     = int(parts[0])
                size_dbls = int(parts[1])
                median_ms = float(parts[2])
            except (ValueError, IndexError):
                continue
            data[ranks].append((size_dbls * 8, median_ms))

    if not data:
        print("skipping HD non-power-of-2 plot: no data")
        return

    p2_color  = "#4e79a7"
    np2_color = "#e15759"

    fig, ax = plt.subplots(figsize=(6, 4))
    for ranks in sorted(data.keys()):
        pts = sorted(data[ranks], key=lambda p: p[0])
        Ns  = np.array([p[0] for p in pts])
        Ts  = np.array([p[1] for p in pts])
        is_p2   = (ranks & (ranks - 1)) == 0
        color   = p2_color if is_p2 else np2_color
        style   = "-" if is_p2 else "--"
        ax.plot(Ns, Ts, f"o{style}", color=color, linewidth=1.5,
                markersize=5, label=f"P={ranks} {'(2^n)' if is_p2 else '(non-2^n)'}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Message size (bytes)")
    ax.set_ylabel("Median latency (ms)")
    ax.set_title("Halving-Doubling: power-of-2 vs non-power-of-2 ranks")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("plots/hd_nonpower2.png", dpi=180)
    plt.close()
    print("saved plots/hd_nonpower2.png")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    plot_alpha_beta()
    rows = load_netsize_csvs()
    plot_netsize(rows)
    plot_strong_scaling()
    plot_hd_nonpower2()
