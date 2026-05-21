import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from causalnex.structure.dynotears import from_pandas_dynamic

# Resolve repo paths relative to this file so the script works from any CWD.
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / "dynotears_baseline_results.csv"

def get_data_and_edges(n_vars, n_samples):
    np.random.seed(45)
    time = np.arange(0, n_samples)
    
    if n_vars == 4:
        e1 = np.random.normal(0, 1, n_samples)
        e2 = np.zeros(n_samples)
        e3 = np.zeros(n_samples)
        e4 = np.zeros(n_samples)
        for t in range(1, n_samples): e2[t] = 0.3 * e1[t-1] + np.random.normal(0, 0.1)
        for t in range(2, n_samples): e3[t] = 0.6 * e2[t-2] + np.random.normal(0, 0.1)
        for t in range(1, n_samples): e4[t] = 0.4 * e3[t-1] + 0.02 * e4[t-1] + np.random.normal(0, 0.1)
        
        x1 = 10 + 2 * np.sin(2 * np.pi * time / 30) + e1
        x2 = 5 + 0.05 * time + e2
        x3 = 7 + np.sin(2 * np.pi * time / 15) + e3
        x4 = 12 + 0.15 * time + e4
        df = pd.DataFrame({'time': time, 'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4})
        gt = [('x1_lag1', 'x2_lag0'), ('x2_lag2', 'x3_lag0'), ('x3_lag1', 'x4_lag0'), ('x4_lag1', 'x4_lag0')]
        
    elif n_vars == 6:
        e1 = np.random.normal(0, 1, n_samples)
        e2 = np.zeros(n_samples)
        e3 = np.zeros(n_samples)
        e4 = np.zeros(n_samples)
        e5 = np.zeros(n_samples)
        e6 = np.zeros(n_samples)
        for t in range(1, n_samples): e2[t] = 0.7 * e1[t-1] + np.random.normal(0, 0.1)
        for t in range(2, n_samples): e3[t] = 0.6 * e2[t-2] + np.random.normal(0, 0.1)
        for t in range(1, n_samples): e4[t] = 0.5 * e3[t-1] + 0.8 * e4[t-1] + np.random.normal(0, 0.1)
        for t in range(1, n_samples): e5[t] = 0.65 * e4[t] + np.random.normal(0, 0.1)
        for t in range(2, n_samples): e6[t] = 0.75 * e5[t-1] + np.random.normal(0, 0.1)
        
        x1 = 10 + 2 * np.sin(2 * np.pi * time / 30) + e1
        x2 = 5 + 0.05 * time + np.sin(2 * np.pi * time / 45) + e2
        x3 = 7 + np.sin(2 * np.pi * time / 15) + e3
        x4 = 12 + 0.15 * time + e4
        x5 = 10 + e5
        x6 = 8 + 0.1 * time + np.sin(2 * np.pi * time / 12) + e6
        df = pd.DataFrame({'time': time, 'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'x6': x6})
        gt = [('x1_lag1', 'x2_lag0'), ('x2_lag2', 'x3_lag0'), ('x3_lag1', 'x4_lag0'), ('x4_lag1', 'x4_lag0'), ('x4_lag0', 'x5_lag0'), ('x5_lag1', 'x6_lag0')]
        
    elif n_vars == 8:
        e1 = np.random.normal(0, 1, n_samples)
        e2, e3, e4, e5, e6, e7, e8 = [np.zeros(n_samples) for _ in range(7)]
        for t in range(1, n_samples): e2[t] = 0.7 * e1[t-1] + np.random.normal(0, 0.1)
        for t in range(n_samples): e3[t] = 0.8 * e2[t] + np.random.normal(0, 0.1)
        for t in range(2, n_samples): e4[t] = 0.5 * e3[t-2] + np.random.normal(0, 0.1)
        for t in range(1, n_samples): e5[t] = 0.7 * e4[t] + 0.3 * e4[t-1] + np.random.normal(0, 0.1)
        for t in range(2, n_samples): e6[t] = 0.6 * e5[t-2] + np.random.normal(0, 0.1)
        for t in range(n_samples): e7[t] = 0.75 * e6[t] + np.random.normal(0, 0.1)
        for t in range(1, n_samples): e8[t] = 0.8 * e7[t-1] + np.random.normal(0, 0.1)

        x1 = 10 + e1
        x2 = 5 + 0.05 * time + e2
        x3 = 7 + e3
        x4 = 12 + np.sin(2 * np.pi * time / 20) + e4
        x5 = 10 + 0.1 * time + 3 * np.sin(2 * np.pi * time / 25) + e5
        x6 = 8 + 0.2 * time + e6
        x7 = 9 + e7
        x8 = 11 + 4 * np.sin(2 * np.pi * time / 12) + e8
        df = pd.DataFrame({'time': time, 'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'x6': x6, 'x7': x7, 'x8': x8})
        gt = [('x1_lag1', 'x2_lag0'), ('x2_lag0', 'x3_lag0'), ('x3_lag2', 'x4_lag0'), ('x4_lag0', 'x5_lag0'), ('x4_lag1', 'x5_lag0'), ('x5_lag2', 'x6_lag0'), ('x6_lag0', 'x7_lag0'), ('x7_lag1', 'x8_lag0')]
    
    return df, gt

def calculate_metrics(discovered_edges, ground_truth_edges):
    true_positives = len(set(ground_truth_edges) & set(discovered_edges))
    false_positives = len(set(discovered_edges) - set(ground_truth_edges))
    false_negatives = len(set(ground_truth_edges) - set(discovered_edges))
    TPR = true_positives / len(ground_truth_edges) if ground_truth_edges else 0
    FDR = false_positives / len(discovered_edges) if discovered_edges else 0
    SHD = false_positives + false_negatives
    return TPR, FDR, SHD

def main():
    variables_list = [4, 6, 8]
    samples_list = [500, 1000, 1500]
    lags_list = [2, 4, 6]
    
    out = open(RESULTS_PATH, 'w')
    out.write("Variables,Samples,Lag,TPR,FDR,SHD\n")
    print("Variables\tSamples\tLag\tTPR\tFDR\tSHD")
    print("-" * 55)
    
    for n_vars in variables_list:
        for n_samples in samples_list:
            df, ground_truth_edges = get_data_and_edges(n_vars, n_samples)
            data_dyno = df.drop('time', axis=1)
            for lag in lags_list:
                try:
                    sm = from_pandas_dynamic(data_dyno, p=lag, lambda_w=0.05, lambda_a=0.05, w_threshold=0.05)
                    discovered_edges = []
                    for source, targets in sm.adj.items():
                        for target in targets:
                            discovered_edges.append((source, target))
                    tpr, fdr, shd = calculate_metrics(discovered_edges, ground_truth_edges)
                except Exception as e:
                    tpr, fdr, shd = 0.0, 1.0, 99
                    print(f"Error on {n_vars} {n_samples} {lag}: {e}")
                
                log_str = f"{n_vars}\t\t{n_samples}\t{lag}\t{tpr:.3f}\t{fdr:.3f}\t{shd}"
                print(log_str)
                out.write(f"{n_vars},{n_samples},{lag},{tpr:.3f},{fdr:.3f},{shd}\n")
                out.flush()

    out.close()

if __name__ == '__main__':
    main()
