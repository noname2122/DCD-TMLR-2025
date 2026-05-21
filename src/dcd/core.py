"""Core DCD pipeline: STL decomposition + PCMCI+ on residuals.

This module exposes the public API used by the experiment scripts and
notebooks: ``load_dataset``, ``decompose_all``, ``run_pcmci_analysis``,
``plot_decomposition``, plus the lower-level helpers
``find_optimal_stl_period``, ``decompose_series``, ``check_stationarity``,
and ``hsic_test_with_time``.

A command-line entry point (``python -m dcd.core <csv> [options]``) is
provided for quick ad-hoc runs without writing a script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.cmiknn import CMIknn


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(df, time_col=None, feature_cols=None):
    """Return a copy of ``df`` with a time column followed by feature columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    time_col : str or None
        Name of the time column. If ``None``, a sequential index is created.
    feature_cols : list[str] or None
        Columns to keep. If ``None``, all numeric columns (excluding the
        time column) are used.
    """
    if time_col is None or time_col not in df.columns:
        df = df.copy()
        df["time"] = np.arange(len(df))
        time_col = "time"

    if feature_cols is None:
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns if c != time_col
        ]
    else:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns {missing} not found")

    return df[[time_col] + list(feature_cols)]


# ---------------------------------------------------------------------------
# STL decomposition
# ---------------------------------------------------------------------------
def find_optimal_stl_period(series, max_period=50, variance_threshold=0.1):
    """Pick the period maximising the seasonal variance from STL."""
    best_period = None
    best_seasonal_variance = 0.0

    for period in range(2, max_period + 1):
        stl = STL(series, period=period, seasonal=13)
        result = stl.fit()
        seasonal_variance = np.var(result.seasonal)
        if (
            seasonal_variance > best_seasonal_variance
            and seasonal_variance > variance_threshold
        ):
            best_period = period
            best_seasonal_variance = seasonal_variance

    return best_period


def decompose_series(series, period):
    """Return ``(trend, seasonal, residual)`` from STL with sensible defaults."""
    if period is None:
        return series, np.zeros(len(series)), series - series.mean()

    seasonal = 15
    trend = max(int(period) + 1, 35)
    if trend % 2 == 0:
        trend += 1

    stl = STL(series, seasonal=seasonal, trend=trend, period=int(period))
    result = stl.fit()
    return result.trend, result.seasonal, result.resid


def decompose_all(df, time_col="time", max_period=50, variance_threshold=0.1):
    """STL-decompose every non-time column. Returns ``(components_df, periods)``."""
    components = {}
    optimal_periods = {}
    feature_cols = [c for c in df.columns if c != time_col]

    for col in feature_cols:
        period = find_optimal_stl_period(df[col], max_period, variance_threshold)
        optimal_periods[col] = period
        print(f"Using period {period} for {col}")

    for col in feature_cols:
        trend, seasonal, resid = decompose_series(df[col], optimal_periods[col])
        components[f"{col}_trend"] = trend
        components[f"{col}_seasonal"] = seasonal
        components[f"{col}_residual"] = resid

    return pd.DataFrame(components), optimal_periods


# ---------------------------------------------------------------------------
# Component-level inference
# ---------------------------------------------------------------------------
def check_stationarity(series, series_name):
    """ADF + KPSS joint stationarity check. Returns ``True`` if stationary."""
    adf_result = adfuller(series)
    kpss_result = kpss(series, regression="c")
    is_stationary = adf_result[1] < 0.05 and kpss_result[1] > 0.05
    if not is_stationary:
        print(f"t --> {series_name.replace('_trend', '')}")
    return is_stationary


def hsic_test_with_time(
    seasonal_component, time, gamma=0.1, num_permutations=500, variance_threshold=0.01
):
    """Permutation HSIC test between a seasonal component and time."""
    if np.var(seasonal_component) < variance_threshold:
        print(
            f"Skipping HSIC test due to low variance: {np.var(seasonal_component):.4f}"
        )
        return None, None

    K_time = rbf_kernel(time, gamma=gamma)
    L_seasonal = rbf_kernel(seasonal_component.values.reshape(-1, 1), gamma=gamma)
    n = K_time.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    HSIC = (1 / (n - 1) ** 2) * np.trace(K_time @ H @ L_seasonal @ H)

    hsic_perm_values = []
    for _ in range(num_permutations):
        perm = np.random.permutation(n)
        L_perm = L_seasonal[perm][:, perm]
        hsic_perm_values.append((1 / (n - 1) ** 2) * np.trace(K_time @ H @ L_perm @ H))

    p_value = float(np.mean([1 if h > HSIC else 0 for h in hsic_perm_values]))
    return HSIC, p_value


def run_pcmci_analysis(components_df, max_lag=2):
    """PCMCI+ with CMIknn on the residual components of ``components_df``."""
    residual_cols = [c for c in components_df.columns if "residual" in c]
    data = components_df[residual_cols].values

    dataframe = pp.DataFrame(data, var_names=residual_cols)
    cmi = CMIknn(
        significance="shuffle_test",
        knn=0.2,
        shuffle_neighbors=5,
        transform="ranks",
        workers=-1,
    )
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cmi, verbosity=1)

    results = pcmci.run_pcmciplus(tau_min=0, tau_max=max_lag, pc_alpha=0.05)
    pcmci.print_significant_links(
        p_matrix=results["p_matrix"],
        val_matrix=results["val_matrix"],
        alpha_level=0.05,
    )
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_decomposition(data, components_df, time_col="time"):
    """Plot raw / trend / seasonal / residual panels per variable."""
    feature_cols = [c for c in data.columns if c != time_col]

    for column in feature_cols:
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f"Decomposition of {column}")

        axes[0].plot(data[time_col], data[column], label="Raw Data", color="blue")
        axes[0].set_ylabel("Raw Data")
        axes[0].legend()

        axes[1].plot(
            data[time_col],
            components_df[f"{column}_trend"],
            label="Trend",
            color="green",
        )
        axes[1].set_ylabel("Trend")
        axes[1].legend()

        axes[2].plot(
            data[time_col],
            components_df[f"{column}_seasonal"],
            label="Seasonal",
            color="orange",
        )
        axes[2].set_ylabel("Seasonal")
        axes[2].legend()

        axes[3].plot(
            data[time_col],
            components_df[f"{column}_residual"],
            label="Residual",
            color="red",
        )
        axes[3].set_ylabel("Residual")
        axes[3].legend()

        plt.xlabel("Time")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the DCD pipeline (STL decomposition + PCMCI+) on a CSV file."
        )
    )
    parser.add_argument("csv", type=Path, help="Path to the input CSV/Excel file")
    parser.add_argument(
        "--time-col",
        default=None,
        help="Name of the time column (default: create a sequential index)",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Comma-separated list of feature columns (default: all numeric)",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=2,
        help="Maximum lag for PCMCI+ (default: 2)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting the decomposition (useful in headless environments)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.csv.suffix.lower() in {".xls", ".xlsx"}:
        df_raw = pd.read_excel(args.csv)
    else:
        df_raw = pd.read_csv(args.csv)

    print("\nFirst few rows of your data:")
    print(df_raw.head())
    print("\nAvailable columns:", df_raw.columns.tolist())
    print("Dataset shape:", df_raw.shape)

    feature_cols = (
        [c.strip() for c in args.features.split(",")] if args.features else None
    )
    df = load_dataset(df_raw, args.time_col, feature_cols)

    print("\nDecomposing time series...")
    components_df, optimal_periods = decompose_all(df, "time")

    if not args.no_plot:
        print("\nGenerating plots...")
        plot_decomposition(df, components_df, "time")

    print("\nChecking stationarity of trend components:")
    trend_vars = [c.replace("_trend", "") for c in components_df.columns if "trend" in c]
    for var in trend_vars:
        check_stationarity(components_df[f"{var}_trend"], f"{var}_trend")

    print("\nRunning PCMCI+ analysis...")
    pcmci_results = run_pcmci_analysis(components_df, args.max_lag)

    return df, components_df, pcmci_results, optimal_periods


if __name__ == "__main__":
    main()
