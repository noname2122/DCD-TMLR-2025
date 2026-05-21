# DCD — Decomposition-Based Causal Discovery in Time Series

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

This repository contains the reference implementation and experiment scripts for **DCD (Decomposition-Based Causal Discovery)**, a framework for recovering causal structure from non-stationary, seasonal time series. It also includes head-to-head baselines (DYNOTEARS, PCMCI+, CD-NOD) and the synthetic / real-world datasets used in the TMLR paper.

> *This repository is maintained anonymously for double-blind peer review. Author details will be disclosed post-review.*

---

## 1. What problem does DCD solve?

Causal discovery methods for time series tend to break down when the data is dominated by periodic or seasonal components. Strong auto-correlations from seasonality create spurious "causal" edges, and as the lag horizon `τ_max` grows the false discovery rate explodes.

DCD fixes this with a simple two-step recipe:

1. **Decompose** each variable with Seasonal-Trend (STL) decomposition and keep the residual component.
2. **Discover** on the residuals with PCMCI+ (CMI-knn independence test).

This preserves the true lagged dependence structure while removing the cyclic components that confound standard algorithms, so DCD keeps high TPR and low SHD well beyond `τ_max = 4`.

## 2. Repository layout

```
DCD-TMLR-2025/
├── src/
│   └── dcd/                       # installable Python package
│       ├── __init__.py
│       ├── core.py                # DCD pipeline: STL + PCMCI+
│       ├── baselines/
│       │   └── cdnod_pcmci.py     # CD-NOD and PCMCI+ baselines on Arctic data
│       └── utils/
│           └── extract_code.py    # helper: notebook -> .py
├── experiments/                   # runnable experiment drivers
│   ├── ablation.py
│   ├── extensive_ablation.py
│   ├── plot_arctic.py
│   ├── plot_ehh1.py
│   ├── dynotears_arctic.py
│   └── run_dynotears_baseline.py
├── scripts/
│   └── regenerate_paper_tables.py # rebuilds the summary tables in results/
├── notebooks/                     # annotated walkthroughs
│   ├── DCD.ipynb
│   ├── DYNOTEARS.ipynb
│   ├── CD_NOD_PCMCI+_on_SIE.ipynb
│   └── Synthetic_data_results.ipynb
├── datasets/
│   ├── Arctic_Monthly.csv         # real-world climate dataset
│   ├── ehh1.csv                   # real-world (EHH-1) dataset
│   ├── lag_2/                     # synthetic data, ground-truth max lag = 2
│   ├── lag_3/                     # ground-truth max lag = 3
│   └── lag_4/                     # ground-truth max lag = 4
├── replication/
│   └── isolation_test.py          # Table A3 isolation-test replication script
├── results/                       # CSV outputs from experiments
├── figures/                       # generated plots
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

Each synthetic CSV is named `{n_vars}.{lag}.{n_samples}.csv`, e.g. `6.3.1000.csv` is 6 variables, ground-truth lag 3, 1000 time steps.

## 3. Installation

Tested on Python 3.8 – 3.11.

```bash
git clone https://github.com/noname2122/DCD-TMLR-2025.git
cd DCD-TMLR-2025

# Option A: just install dependencies
pip install -r requirements.txt

# Option B: editable install of the `dcd` package
pip install -e .
```

The heaviest dependencies are [`tigramite`](https://github.com/jakobrunge/tigramite),
[`causalnex`](https://github.com/mckinsey/causalnex), and
[`causal-learn`](https://github.com/py-why/causal-learn). We recommend a fresh
virtual environment.

## 4. Reproducing the paper

All experiment drivers live under `experiments/` and can be run directly —
they resolve dataset/result paths relative to the repo root, so you can
launch them from anywhere.

### 4.1 Ablation on synthetic data (Tables 2–4)
```bash
python experiments/extensive_ablation.py
```
Sweeps `n_vars ∈ {4,6,8}`, `n_samples ∈ {500,1000,1500}`,
`lag ∈ {2,4,6}`, `period ∈ {10,15,20,25,30,35}` and writes
`results/extensive_ablation_results.csv`.

### 4.2 DYNOTEARS baseline on the same synthetic grid
```bash
python experiments/run_dynotears_baseline.py
```
Writes `results/dynotears_baseline_results.csv`.

### 4.3 Real-world case studies (Arctic climate, EHH-1)
```bash
python experiments/plot_arctic.py       # → figures/arctic_dynotears_graph.png
python experiments/plot_ehh1.py         # → figures/ehh1_dynotears_graph.png
python experiments/dynotears_arctic.py  # prints edge list for Arctic_Monthly.csv
```

### 4.4 PCMCI+ / CD-NOD on Arctic
```bash
python -m dcd.baselines.cdnod_pcmci --method pcmci
python -m dcd.baselines.cdnod_pcmci --method cdnod
```

### 4.5 Regenerating the paper summary tables
```bash
python scripts/regenerate_paper_tables.py
```
Writes `algorithm_comparison.csv`, `statistics_by_sample_size.csv`,
and `statistics_by_i_value.csv` to `results/`.

### 4.6 Isolation test (Table A3)
```bash
python replication/isolation_test.py
```
Runs the multi-scale isolation test (d=6, n=1000, 3 seeds) and writes
`replication/isolation_comparison.csv` comparing PCMCI+, DYNOTEARS, and DCD
on raw / STL-residual / multi-scale inputs.

### 4.7 Interactive walkthroughs
Open the notebooks for step-by-step versions of each pipeline:
- `notebooks/DCD.ipynb`
- `notebooks/DYNOTEARS.ipynb`
- `notebooks/CD_NOD_PCMCI+_on_SIE.ipynb`
- `notebooks/Synthetic_data_results.ipynb`

## 5. Using DCD on your own data

As a library:

```python
import pandas as pd
from dcd.core import load_dataset, decompose_all, run_pcmci_analysis

df = load_dataset(pd.read_csv("your_series.csv"), time_col="time")
components_df, periods = decompose_all(df, "time")
results = run_pcmci_analysis(components_df, max_lag=4)
```

Or from the command line:

```bash
python -m dcd.core your_series.csv --time-col time --max-lag 4 --no-plot
```

## 6. Summary of findings

1. DCD maintains **TPR ≈ 1.0** across lag depths up to 6, where baselines drop to ≤ 0.3.
2. DYNOTEARS produces **> 90% false-discovery edges** on periodic systems at `τ_max > 2`.
3. STL decomposition prior to independence-based discovery is necessary for reliable recovery in multivariate periodic time series of up to 1500 samples.

## 7. Paper

The TMLR submission manuscript is not redistributed in this repository during the review period.

## 8. License

Released under the MIT License — see [`LICENSE`](LICENSE).

## 9. Citation

BibTeX will be added after peer review.
