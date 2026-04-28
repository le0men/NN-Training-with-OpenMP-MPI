#!/usr/bin/env python3
"""
Fit the alpha-beta latency model (T = alpha + beta * N) to allreduce_bench output.

Usage:
    mpirun -np 8 ./build/allreduce_bench | python3 scripts/fit_alpha_beta.py
    python3 scripts/fit_alpha_beta.py bench_output.txt
    python3 scripts/fit_alpha_beta.py bench_output.txt --plot

The alpha-beta model:
    T(N) = alpha + beta * N * sizeof(double)
    alpha = per-message latency (us)
    beta  = inverse bandwidth (us/byte) -> 1/beta = bandwidth (GB/s)

allreduce_bench prints:
    size   algo   median_ms   min_ms   GB/s_eff
where `size` is the number of doubles in the buffer.
"""

import sys
import re
import numpy as np
from collections import defaultdict

SIZEOF_DOUBLE = 8  # bytes


def parse_bench_output(lines):
    """Return dict: algo -> list of (N_bytes, median_s)."""
    data = defaultdict(list)
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            n_doubles = int(parts[0])
            algo = parts[1]
            median_ms = float(parts[2])
        except ValueError:
            continue
        n_bytes = n_doubles * SIZEOF_DOUBLE
        median_s = median_ms / 1000.0
        data[algo].append((n_bytes, median_s))
    return data


def fit_alpha_beta(points):
    """
    Fit T = alpha + beta * N via least-squares linear regression.
    points: list of (N_bytes, T_seconds)
    Returns (alpha_us, beta_us_per_byte, r_squared)
    """
    if len(points) < 2:
        return None, None, None
    Ns = np.array([p[0] for p in points], dtype=float)
    Ts = np.array([p[1] for p in points], dtype=float)
    # Fit degree-1 polynomial: T = beta * N + alpha
    coeffs = np.polyfit(Ns, Ts, 1)
    beta_s_per_byte = coeffs[0]
    alpha_s = coeffs[1]
    # R^2
    T_pred = np.polyval(coeffs, Ns)
    ss_res = np.sum((Ts - T_pred) ** 2)
    ss_tot = np.sum((Ts - Ts.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return alpha_s * 1e6, beta_s_per_byte * 1e6, r2  # convert to us


def print_results(data, num_ranks):
    algo_order = ["tree", "ring", "halving_doubling", "mpi", "builtin"]
    algos = sorted(data.keys(), key=lambda a: algo_order.index(a) if a in algo_order else 99)

    print(f"\n{'='*72}")
    print(f"  alpha-beta model fit  (P = {num_ranks} ranks)")
    print(f"  T(N) = alpha + beta * N    where N = message size in bytes")
    print(f"{'='*72}")
    header = f"  {'algo':<18} {'alpha (us)':>12} {'beta (us/B)':>12} {'BW (GB/s)':>12} {'R^2':>8}"
    print(header)
    print(f"  {'-'*68}")

    for algo in algos:
        pts = data[algo]
        alpha_us, beta_us_per_byte, r2 = fit_alpha_beta(pts)
        if alpha_us is None:
            print(f"  {algo:<18}   (insufficient data points)")
            continue
        bw_gbs = 1.0 / (beta_us_per_byte * 1e-6) / 1e9 if beta_us_per_byte > 0 else float("inf")
        print(f"  {algo:<18} {alpha_us:>12.2f} {beta_us_per_byte:>12.4f} {bw_gbs:>12.3f} {r2:>8.4f}")

    print(f"{'='*72}")
    print()
    print("  Interpretation:")
    print("    alpha  -- per-call startup latency (microseconds)")
    print("    beta   -- marginal cost per byte (us/byte);  1/beta = peak BW")
    print("    R^2    -- goodness of fit (1.0 = perfect linear model)")
    print()


def maybe_plot(data, num_ranks, out_path="alpha_beta_fit.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available -- skipping plot")
        return

    algo_order = ["tree", "ring", "halving_doubling", "mpi", "builtin"]
    colors = {"tree": "tab:blue", "ring": "tab:orange",
              "halving_doubling": "tab:green", "mpi": "tab:red", "builtin": "tab:red"}
    algos = sorted(data.keys(), key=lambda a: algo_order.index(a) if a in algo_order else 99)

    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in algos:
        pts = data[algo]
        Ns = np.array([p[0] for p in pts])
        Ts = np.array([p[1] for p in pts]) * 1e6  # seconds -> us
        color = colors.get(algo, None)
        ax.scatter(Ns, Ts, label=algo, color=color, zorder=3)

        alpha_us, beta_us, r2 = fit_alpha_beta(pts)
        if alpha_us is not None:
            Nfit = np.linspace(Ns.min(), Ns.max(), 200)
            Tfit = alpha_us + beta_us * Nfit
            ax.plot(Nfit, Tfit, "--", color=color, linewidth=1)

    ax.set_xlabel("Message size (bytes)")
    ax.set_ylabel("Median latency (us)")
    ax.set_title(f"alpha-beta fit  (P={num_ranks})")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"  Plot saved to {out_path}")


def extract_num_ranks(lines):
    for line in lines:
        m = re.search(r"(\d+)\s+rank", line)
        if m:
            return int(m.group(1))
    return "?"


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", help="bench output file (default: stdin)")
    parser.add_argument("--plot", action="store_true", help="save a PNG of the fits")
    parser.add_argument("--plot-out", default="alpha_beta_fit.png",
                        help="output path for the plot (default: alpha_beta_fit.png)")
    args = parser.parse_args()

    if args.input:
        with open(args.input) as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    num_ranks = extract_num_ranks(lines)
    data = parse_bench_output(lines)

    if not data:
        print("No data found — make sure you pass allreduce_bench output.", file=sys.stderr)
        sys.exit(1)

    print_results(data, num_ranks)

    if args.plot:
        maybe_plot(data, num_ranks, args.plot_out)


if __name__ == "__main__":
    main()
