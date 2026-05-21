"""Replicate DCD on the saved CSV datasets and compare against paper-reported numbers.

For each CSV in datasets/lag_{2,3,4}/ the script:
  1. Infers (n_vars, gt_lag, n_samples) from the filename.
  2. STL-decomposes every variable (auto-detects the seasonal period).
  3. Runs PCMCI+ on the residuals with tau_max = gt_lag.
  4. Computes TPR / FDR / SHD against the known ground-truth causal graph.

Results are written to:
  replication/raw_results.csv          -- one row per dataset file
  replication/comparison_dcd.csv       -- replicated vs paper numbers for DCD

Usage:
  python experiments/replicate_paper_tables.py
  python experiments/replicate_paper_tables.py --vars 4 6  # subset of n_vars
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import periodogram as _fft_periodogram
from statsmodels.tsa.seasonal import STL
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.cmiknn import CMIknn

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "datasets"
REPLICATION_DIR = REPO_ROOT / "replication"
REPLICATION_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Ground-truth causal edges for every (n_vars, gt_lag) configuration.
# Derived from the data-generation notebooks in datasets/lag_{2,3,4}/.
# Format: (source_lagTAU, target_lag0)
# ---------------------------------------------------------------------------
GROUND_TRUTH: dict[tuple[int, int], list[tuple[str, str]]] = {
    # 4-var: x1->x2(lag1), x2->x3(lag2), x3->x4(lagL), x4->x4(lag1)
    (4, 2): [
        ("x1_lag1", "x2_lag0"), ("x2_lag2", "x3_lag0"),
        ("x3_lag1", "x4_lag0"), ("x4_lag1", "x4_lag0"),
    ],
    (4, 3): [
        ("x1_lag1", "x2_lag0"), ("x2_lag2", "x3_lag0"),
        ("x3_lag3", "x4_lag0"), ("x4_lag1", "x4_lag0"),
    ],
    (4, 4): [
        ("x1_lag1", "x2_lag0"), ("x2_lag2", "x3_lag0"),
        ("x3_lag4", "x4_lag0"), ("x4_lag1", "x4_lag0"),
    ],
    # 6-var: extends 4-var base (x4_lag1->x4, x3_lagL->x4) +
    #        x4->x5(lag0), x5->x6(lag1/lag4)
    (6, 2): [
        ("x1_lag1", "x2_lag0"), ("x2_lag2", "x3_lag0"),
        ("x3_lag1", "x4_lag0"), ("x4_lag1", "x4_lag0"),
        ("x4_lag0", "x5_lag0"), ("x5_lag1", "x6_lag0"),
    ],
    (6, 3): [
        ("x1_lag1", "x2_lag0"), ("x2_lag2", "x3_lag0"),
        ("x3_lag3", "x4_lag0"), ("x4_lag1", "x4_lag0"),
        ("x4_lag0", "x5_lag0"), ("x5_lag1", "x6_lag0"),
    ],
    (6, 4): [
        ("x1_lag1", "x2_lag0"), ("x2_lag2", "x3_lag0"),
        ("x3_lag3", "x4_lag0"), ("x4_lag1", "x4_lag0"),
        ("x4_lag0", "x5_lag0"), ("x5_lag4", "x6_lag0"),
    ],
    # 8-var: x1->x2(1), x2->x3(0), x3->x4(2/3), x4->x5(0,1),
    #        x5->x6(2/4), x6->x7(0), x7->x8(1)
    (8, 2): [
        ("x1_lag1", "x2_lag0"), ("x2_lag0", "x3_lag0"),
        ("x3_lag2", "x4_lag0"), ("x4_lag0", "x5_lag0"),
        ("x4_lag1", "x5_lag0"), ("x5_lag2", "x6_lag0"),
        ("x6_lag0", "x7_lag0"), ("x7_lag1", "x8_lag0"),
    ],
    (8, 3): [
        ("x1_lag1", "x2_lag0"), ("x2_lag0", "x3_lag0"),
        ("x3_lag3", "x4_lag0"), ("x4_lag0", "x5_lag0"),
        ("x4_lag1", "x5_lag0"), ("x5_lag2", "x6_lag0"),
        ("x6_lag0", "x7_lag0"), ("x7_lag1", "x8_lag0"),
    ],
    (8, 4): [
        ("x1_lag1", "x2_lag0"), ("x2_lag0", "x3_lag0"),
        ("x3_lag3", "x4_lag0"), ("x4_lag0", "x5_lag0"),
        ("x4_lag1", "x5_lag0"), ("x5_lag4", "x6_lag0"),
        ("x6_lag0", "x7_lag0"), ("x7_lag1", "x8_lag0"),
    ],
    # 10-var: 8-var + x8->x9(lag2), x9->x10(lag1)  (same across all lags)
    (10, 2): [
        ("x1_lag1", "x2_lag0"), ("x2_lag0", "x3_lag0"),
        ("x3_lag2", "x4_lag0"), ("x4_lag0", "x5_lag0"),
        ("x4_lag1", "x5_lag0"), ("x5_lag2", "x6_lag0"),
        ("x6_lag0", "x7_lag0"), ("x7_lag1", "x8_lag0"),
        ("x8_lag2", "x9_lag0"), ("x9_lag1", "x10_lag0"),
    ],
    (10, 3): [
        ("x1_lag1", "x2_lag0"), ("x2_lag0", "x3_lag0"),
        ("x3_lag3", "x4_lag0"), ("x4_lag0", "x5_lag0"),
        ("x4_lag1", "x5_lag0"), ("x5_lag2", "x6_lag0"),
        ("x6_lag0", "x7_lag0"), ("x7_lag1", "x8_lag0"),
        ("x8_lag2", "x9_lag0"), ("x9_lag1", "x10_lag0"),
    ],
    (10, 4): [
        ("x1_lag1", "x2_lag0"), ("x2_lag0", "x3_lag0"),
        ("x3_lag3", "x4_lag0"), ("x4_lag0", "x5_lag0"),
        ("x4_lag1", "x5_lag0"), ("x5_lag4", "x6_lag0"),
        ("x6_lag0", "x7_lag0"), ("x7_lag1", "x8_lag0"),
        ("x8_lag2", "x9_lag0"), ("x9_lag1", "x10_lag0"),
    ],
}

# DCD numbers from the paper (scripts/regenerate_paper_tables.py DATA dict).
# Keyed by (gt_lag, n_vars, n_samples).
PAPER_DCD: dict[tuple[int, int, int], list[float]] = {
    (2, 4,  500): [0.889, 0.0,   1],
    (2, 4, 1000): [0.78,  0.125, 3],
    (2, 4, 1500): [0.78,  0.125, 3],
    (2, 6,  500): [0.769, 0.286, 7],
    (2, 6, 1000): [0.769, 0.286, 7],
    (2, 6, 1500): [0.769, 0.286, 7],
    (2, 8,  500): [0.786, 0.389, 10],
    (2, 8, 1000): [0.786, 0.353, 9],
    (2, 8, 1500): [0.786, 0.389, 10],
    (2, 10,  500): [0.765, 0.381, 12],
    (2, 10, 1000): [0.765, 0.381, 12],
    (2, 10, 1500): [0.765, 0.409, 13],
    (3, 4,  500): [0.8,   0.11,  3],
    (3, 4, 1000): [0.9,   0.1,   2],
    (3, 4, 1500): [0.8,   0.2,   4],
    (3, 6,  500): [0.769, 0.412, 10],
    (3, 6, 1000): [0.769, 0.444, 11],
    (3, 6, 1500): [0.769, 0.412, 10],
    (3, 8,  500): [0.786, 0.45,  12],
    (3, 8, 1000): [0.786, 0.476, 13],
    (3, 8, 1500): [0.786, 0.5,   14],
    (3, 10,  500): [0.765, 0.458, 15],
    (3, 10, 1000): [0.765, 0.458, 15],
    (3, 10, 1500): [0.765, 0.48,  16],
    (4, 4,  500): [0.8,   0.385, 7],
    (4, 4, 1000): [0.9,   0.4,   7],
    (4, 4, 1500): [0.9,   0.4,   7],
    (4, 6,  500): [0.769, 0.545, 15],
    (4, 6, 1000): [0.769, 0.545, 15],
    (4, 6, 1500): [0.769, 0.545, 15],
    (4, 8,  500): [0.786, 0.5,   14],
    (4, 8, 1000): [0.786, 0.522, 15],
    (4, 8, 1500): [0.786, 0.522, 15],
    (4, 10,  500): [0.765, 0.458, 15],
    (4, 10, 1000): [0.765, 0.458, 15],
    (4, 10, 1500): [0.765, 0.48,  16],
}


# ---------------------------------------------------------------------------
# STL helpers (self-contained, no import from dcd.core)
# ---------------------------------------------------------------------------

def find_optimal_period(series: pd.Series, max_period: int = 40,
                        power_threshold: float = 0.05) -> int | None:
    """Return the dominant seasonal period from a Welch periodogram.

    Detrends the series linearly, then looks for the peak frequency in the
    range [4, max_period].  Returns None if no frequency explains at least
    `power_threshold` fraction of total spectral power.
    """
    x = series.values.astype(float)
    n = len(x)
    t = np.arange(n, dtype=float)
    x = x - np.polyval(np.polyfit(t, x, 1), t)

    freq, power = _fft_periodogram(x)
    total = float(np.sum(power[freq > 0]))
    if total == 0:
        return None

    mask = (freq > 0) & (1.0 / freq >= 4) & (1.0 / freq <= max_period)
    if not mask.any():
        return None

    best_idx = int(np.argmax(power[mask]))
    best_freq = freq[mask][best_idx]
    best_power = float(power[mask][best_idx])

    if best_power / total < power_threshold:
        return None

    period = int(round(1.0 / best_freq))
    return max(period, 4)


def decompose_residual(series: pd.Series, period: int | None) -> np.ndarray:
    t = np.arange(len(series), dtype=float)
    if period is None:
        # No seasonal component: remove polynomial trend (degree 1) only.
        coeffs = np.polyfit(t, series.values, 1)
        return series.values - np.polyval(coeffs, t)
    seasonal = 15
    trend_win = max(int(period) + 1, 35)
    if trend_win % 2 == 0:
        trend_win += 1
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1e-10, len(series))
    stl = STL(series + noise, seasonal=seasonal, trend=trend_win, period=int(period))
    return stl.fit().resid


def stl_decompose_df(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c != time_col]
    residuals = {}
    for col in feature_cols:
        period = find_optimal_period(df[col])
        print(f"  {col}: period={period}", flush=True)
        residuals[f"{col}_residual"] = decompose_residual(df[col], period)
    return pd.DataFrame(residuals)


# ---------------------------------------------------------------------------
# PCMCI+ helper
# ---------------------------------------------------------------------------

def run_pcmci(df_resid: pd.DataFrame, tau_max: int) -> list[tuple[str, str]]:
    residual_cols = [c for c in df_resid.columns if "_residual" in c]
    data = df_resid[residual_cols].values
    var_names = [c.replace("_residual", "") for c in residual_cols]

    dataframe = pp.DataFrame(data, var_names=var_names)
    cmi = CMIknn(
        significance="shuffle_test",
        knn=0.1,
        shuffle_neighbors=3,
        transform="ranks",
        workers=-1,
    )
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cmi, verbosity=0)
    results = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=0.05)

    discovered: list[tuple[str, str]] = []
    p_matrix = results["p_matrix"]
    graph = results.get("graph")

    for i, v1 in enumerate(var_names):
        for j, v2 in enumerate(var_names):
            for tau in range(tau_max + 1):
                if i == j and tau == 0:
                    continue
                if graph is not None:
                    if graph[i, j, tau] in ("-->", "o-o", "x-x"):
                        discovered.append((f"{v1}_lag{tau}", f"{v2}_lag0"))
                else:
                    if p_matrix[i, j, tau] < 0.05:
                        discovered.append((f"{v1}_lag{tau}", f"{v2}_lag0"))
    return list(set(discovered))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    discovered: list[tuple[str, str]],
    ground_truth: list[tuple[str, str]],
) -> tuple[float, float, int]:
    gt = set(ground_truth)
    disc = set(discovered)
    tp = len(gt & disc)
    fp = len(disc - gt)
    fn = len(gt - disc)
    tpr = tp / len(gt) if gt else 0.0
    fdr = fp / len(disc) if disc else 0.0
    shd = fp + fn
    return tpr, fdr, shd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_csv_paths(vars_filter: list[int] | None) -> list[Path]:
    entries: list[tuple[int, int, int, Path]] = []
    for lag_dir in DATASETS_DIR.glob("lag_*"):
        for csv_path in lag_dir.glob("*.csv"):
            parts = csv_path.stem.split(".")
            if len(parts) != 3:
                continue
            n_vars, gt_lag, n_samples = int(parts[0]), int(parts[1]), int(parts[2])
            if vars_filter and n_vars not in vars_filter:
                continue
            entries.append((n_vars, n_samples, gt_lag, csv_path))
    # Sort: smaller n_vars first, then smaller n_samples, then smaller lag.
    entries.sort(key=lambda e: (e[0], e[1], e[2]))
    return [e[3] for e in entries]


def main() -> None:
    parser = argparse.ArgumentParser(description="Replicate DCD on saved CSV datasets")
    parser.add_argument(
        "--vars",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="Restrict to these n_vars values (e.g. --vars 4 6 8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List configs only; do not run any experiments",
    )
    args = parser.parse_args()

    csv_paths = collect_csv_paths(args.vars)
    if not csv_paths:
        sys.exit("No matching CSV files found.")

    print(f"Found {len(csv_paths)} CSV file(s) to process.\n", flush=True)

    if args.dry_run:
        for p in csv_paths:
            parts = p.stem.split(".")
            n_vars, gt_lag, n_samples = int(parts[0]), int(parts[1]), int(parts[2])
            print(f"  {p.relative_to(REPO_ROOT)}  vars={n_vars}  lag={gt_lag}  n={n_samples}")
        return

    raw_path = REPLICATION_DIR / "raw_results.csv"
    raw_fh = open(raw_path, "w")
    raw_fh.write("n_vars,gt_lag,n_samples,TPR,FDR,SHD,n_discovered,n_gt\n")

    rows = []

    for csv_path in csv_paths:
        parts = csv_path.stem.split(".")
        n_vars, gt_lag, n_samples = int(parts[0]), int(parts[1]), int(parts[2])
        gt_edges = GROUND_TRUTH.get((n_vars, gt_lag))
        if gt_edges is None:
            print(f"[SKIP] No ground truth for ({n_vars}, {gt_lag}): {csv_path.name}")
            continue

        label = f"vars={n_vars}  lag={gt_lag}  n={n_samples}"
        print(f"\n{'='*60}\n{label}\n{'='*60}", flush=True)

        df = pd.read_csv(csv_path)
        print(f"  Loaded {df.shape[0]} rows × {df.shape[1]-1} features", flush=True)

        print("  STL decomposition:", flush=True)
        df_resid = stl_decompose_df(df, time_col="time")

        print(f"  Running PCMCI+ (tau_max={gt_lag})…", flush=True)
        discovered = run_pcmci(df_resid, tau_max=gt_lag)

        tpr, fdr, shd = compute_metrics(discovered, gt_edges)
        print(
            f"  TPR={tpr:.3f}  FDR={fdr:.3f}  SHD={shd}  "
            f"(discovered={len(discovered)}, GT={len(gt_edges)})",
            flush=True,
        )

        raw_fh.write(f"{n_vars},{gt_lag},{n_samples},{tpr:.6f},{fdr:.6f},{shd},{len(discovered)},{len(gt_edges)}\n")
        raw_fh.flush()

        rows.append(
            dict(
                n_vars=n_vars,
                gt_lag=gt_lag,
                n_samples=n_samples,
                TPR=tpr,
                FDR=fdr,
                SHD=shd,
            )
        )

    raw_fh.close()
    print(f"\nRaw results → {raw_path}", flush=True)

    if not rows:
        return

    # ------------------------------------------------------------------
    # Build comparison table: replicated vs paper
    # ------------------------------------------------------------------
    records = []
    for row in rows:
        key = (row["gt_lag"], row["n_vars"], row["n_samples"])
        paper = PAPER_DCD.get(key)
        records.append(
            {
                "gt_lag": row["gt_lag"],
                "n_vars": row["n_vars"],
                "n_samples": row["n_samples"],
                "rep_TPR": round(row["TPR"], 4),
                "rep_FDR": round(row["FDR"], 4),
                "rep_SHD": row["SHD"],
                "paper_TPR": paper[0] if paper else None,
                "paper_FDR": round(paper[1], 3) if paper else None,
                "paper_SHD": paper[2] if paper else None,
                "ΔTPR": round(row["TPR"] - paper[0], 4) if paper else None,
                "ΔFDR": round(row["FDR"] - paper[1], 4) if paper else None,
                "ΔSHD": (row["SHD"] - paper[2]) if paper else None,
            }
        )

    cmp_df = pd.DataFrame(records).sort_values(["gt_lag", "n_vars", "n_samples"])
    cmp_path = REPLICATION_DIR / "comparison_dcd.csv"
    cmp_df.to_csv(cmp_path, index=False)
    print(f"Comparison table → {cmp_path}\n", flush=True)

    # Pretty-print summary
    print("\n" + "="*90)
    print(f"{'lag':>4}  {'d':>4}  {'n':>5}  "
          f"{'rep_TPR':>8}  {'ppr_TPR':>8}  "
          f"{'rep_FDR':>8}  {'ppr_FDR':>8}  "
          f"{'rep_SHD':>7}  {'ppr_SHD':>7}  "
          f"{'ΔTPR':>8}  {'ΔFDR':>8}  {'ΔSHD':>6}")
    print("-"*90)
    for _, r in cmp_df.iterrows():
        print(
            f"{r.gt_lag:>4}  {r.n_vars:>4}  {r.n_samples:>5}  "
            f"{r.rep_TPR:>8.3f}  {r.paper_TPR!s:>8}  "
            f"{r.rep_FDR:>8.3f}  {r.paper_FDR!s:>8}  "
            f"{r.rep_SHD:>7}  {r.paper_SHD!s:>7}  "
            f"{r.ΔTPR!s:>8}  {r.ΔFDR!s:>8}  {r.ΔSHD!s:>6}"
        )
    print("="*90)


if __name__ == "__main__":
    main()
