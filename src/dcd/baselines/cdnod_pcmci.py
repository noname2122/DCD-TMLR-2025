"""CD-NOD and PCMCI+ reference baselines on the Arctic Monthly dataset.

Run from the repo root:

    python -m dcd.baselines.cdnod_pcmci --method cdnod
    python -m dcd.baselines.cdnod_pcmci --method pcmci

Importing this module does *not* trigger any computation — the work is
performed inside :func:`run_cdnod` / :func:`run_pcmci` / :func:`main`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Resolve repo paths relative to this file so scripts work from any CWD.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = REPO_ROOT / "datasets" / "Arctic_Monthly.csv"

ARCTIC_FEATURES = [
    "wind_10m",
    "specific_humidity",
    "LW_down",
    "SW_down",
    "rainfall",
    "snowfall",
    "sst",
    "t2m",
    "surface_pressure",
    "sea_ice_extent",
]


def _load_arctic(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def run_cdnod(data_path: Path = DEFAULT_DATA_PATH, alpha: float = 0.05) -> None:
    """Run CD-NOD on the Arctic Monthly dataset and draw the resulting graph."""
    from causallearn.search.ConstraintBased.CDNOD import cdnod
    from causallearn.utils.cit import kci

    df = _load_arctic(data_path)
    c_indx = np.reshape(np.arange(len(df)), (len(df), 1))
    data = df[ARCTIC_FEATURES].values

    cg = cdnod(
        data=data,
        c_indx=c_indx,
        alpha=alpha,
        indep_test=kci,
        verbose=True,
        stable=True,
        uc_rule=0,
        uc_priority=-1,
    )

    print("\nCausal Graph for Arctic Climate Variables:")
    cg.draw_pydot_graph()

    print("\nSummary Statistics:")
    print(df[ARCTIC_FEATURES].describe())

    if "Date" in df.columns:
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(ARCTIC_FEATURES, 1):
            plt.subplot(5, 2, i)
            plt.plot(df["Date"], df[feature])
            plt.title(feature)
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def run_pcmci(
    data_path: Path = DEFAULT_DATA_PATH,
    tau_max: int = 3,
    pc_alpha: float = 0.05,
) -> dict:
    """Run PCMCI+ with CMIknn on Arctic Monthly. Returns the tigramite results dict."""
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.cmiknn import CMIknn

    df = _load_arctic(data_path)
    data = df[ARCTIC_FEATURES].values
    dataframe = pp.DataFrame(data, var_names=ARCTIC_FEATURES)

    cmi = CMIknn(
        significance="shuffle_test",
        knn=0.2,
        shuffle_neighbors=5,
        transform="ranks",
        workers=-1,
    )
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cmi, verbosity=1)
    results = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha)

    pcmci.print_significant_links(
        p_matrix=results["p_matrix"],
        val_matrix=results["val_matrix"],
        alpha_level=pc_alpha,
    )

    print("\nDetailed causal relationships:")
    for i, var1 in enumerate(ARCTIC_FEATURES):
        for j, var2 in enumerate(ARCTIC_FEATURES):
            for tau in range(tau_max + 1):
                if results["p_matrix"][i, j, tau] < pc_alpha:
                    print(
                        f"{var2} -{tau}-> {var1} | "
                        f"p-value = {results['p_matrix'][i, j, tau]:.4f}"
                    )

    n_links = int((results["p_matrix"] < pc_alpha).sum())
    print(f"\nTotal number of significant causal links: {n_links}")
    return results


def plot_arctic_summary_graph() -> None:
    """Plot the hand-curated summary graph reported in the paper for Arctic."""
    import numpy as np
    from tigramite.plotting import plot_graph

    var_names = ARCTIC_FEATURES
    n = len(var_names)
    graph = np.zeros((n, n, 2), dtype="bool")
    val_matrix = np.zeros((n, n, 2))
    link_width = np.zeros((n, n, 2))

    name_to_idx = {name: i for i, name in enumerate(var_names)}

    undirected_links = [
        ("specific_humidity", "LW_down", 0.026),
        ("specific_humidity", "t2m", 0.007),
        ("LW_down", "t2m", 0.012),
        ("snowfall", "surface_pressure", 0.002),
    ]
    directed_contemp_links = [
        ("surface_pressure", "rainfall", 0.022),
        ("wind_10m", "snowfall", 0.034),
        ("wind_10m", "surface_pressure", 0.013),
        ("sst", "sea_ice_extent", 0.016),
    ]
    lagged_links = [
        ("SW_down", "wind_10m", 0.023),
        ("sst", "sst", 0.021),
        ("SW_down", "t2m", 0.010),
        ("surface_pressure", "surface_pressure", 0.023),
    ]

    for i, j, val in undirected_links:
        a, b = name_to_idx[i], name_to_idx[j]
        graph[a, b, 0] = graph[b, a, 0] = True
        val_matrix[a, b, 0] = val_matrix[b, a, 0] = val
        link_width[a, b, 0] = link_width[b, a, 0] = 2.0
    for i, j, val in directed_contemp_links:
        a, b = name_to_idx[i], name_to_idx[j]
        graph[a, b, 0] = True
        val_matrix[a, b, 0] = val_matrix[b, a, 0] = val
        link_width[a, b, 0] = link_width[b, a, 0] = 2.0
    for i, j, val in lagged_links:
        a, b = name_to_idx[i], name_to_idx[j]
        graph[a, b, 1] = True
        val_matrix[a, b, 1] = val
        link_width[a, b, 1] = 2.0

    plot_graph(
        graph=graph,
        val_matrix=val_matrix,
        var_names=var_names,
        link_width=link_width,
        figsize=(10, 6),
        node_size=0.3,
        link_colorbar_label="Link Strength",
        arrow_linewidth=3.0,
    )
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Run CD-NOD or PCMCI+ on the Arctic Monthly dataset."
    )
    parser.add_argument(
        "--method",
        choices=["cdnod", "pcmci", "summary"],
        default="pcmci",
        help="Which baseline to run (default: pcmci).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to Arctic_Monthly.csv (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--tau-max",
        type=int,
        default=3,
        help="Maximum lag for PCMCI+ (default: 3).",
    )
    args = parser.parse_args()

    if args.method == "cdnod":
        run_cdnod(args.data_path)
    elif args.method == "pcmci":
        run_pcmci(args.data_path, tau_max=args.tau_max)
    else:
        plot_arctic_summary_graph()


if __name__ == "__main__":
    main()
