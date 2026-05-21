import sys
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.cmiknn import CMIknn

def generate_synthetic_data(n_samples=500):
    np.random.seed(45)
    time = np.arange(0, n_samples)
    e1 = np.random.normal(0, 1, n_samples)
    e2 = np.zeros(n_samples)
    e3 = np.zeros(n_samples)
    e4 = np.zeros(n_samples)
    e5 = np.zeros(n_samples)
    e6 = np.zeros(n_samples)

    for t in range(1, n_samples):
        e2[t] = 0.7 * e1[t-1] + np.random.normal(0, 0.1)
    for t in range(2, n_samples):
        e3[t] = 0.6 * e2[t-2] + np.random.normal(0, 0.1)
    for t in range(1, n_samples):
        e4[t] = 0.5 * e3[t-1] + 0.8 * e4[t-1] + np.random.normal(0, 0.1)
    for t in range(1, n_samples):
        e5[t] = 0.65 * e4[t] + np.random.normal(0, 0.1)
    for t in range(2, n_samples):
        e6[t] = 0.75 * e5[t-1] + np.random.normal(0, 0.1)

    x1 = 10 + 2 * np.sin(2 * np.pi * time / 30) + e1
    x2 = 5 + 0.05 * time + np.sin(2 * np.pi * time / 45) + e2
    x3 = 7 + np.sin(2 * np.pi * time / 15) + e3
    x4 = 12 + 0.15 * time + e4
    x5 = 10 + e5
    x6 = 8 + 0.1 * time + np.sin(2 * np.pi * time / 12) + e6

    df = pd.DataFrame({'time': time, 'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'x6': x6})
    return df

ground_truth_edges = [
    ('x1_lag1', 'x2_lag0'),
    ('x2_lag2', 'x3_lag0'),
    ('x3_lag1', 'x4_lag0'),
    ('x4_lag1', 'x4_lag0'),
    ('x4_lag0', 'x5_lag0'),
    ('x5_lag1', 'x6_lag0')
]

def decompose_series(series, period):
    if period is None or period <= 2:
        return series - series.mean()
    seasonal = 15
    trend = max(int(period) + 1, 35)
    if trend % 2 == 0:
        trend += 1
    stl = STL(series, seasonal=seasonal, trend=trend, period=int(period))
    result = stl.fit()
    return result.resid

def decompose_all_fixed(df, period):
    components = {}
    time_col = 'time'
    feature_cols = [col for col in df.columns if col != time_col]
    for col in feature_cols:
        resid = decompose_series(df[col], period)
        components[f'{col}_residual'] = resid
    return pd.DataFrame(components)

def df_to_pcmci(df_resid, tau_max=2):
    residual_cols = [col for col in df_resid.columns if 'residual' in col]
    data = df_resid[residual_cols].values
    clean_cols = [col.replace('_residual', '') for col in residual_cols]
    dataframe = pp.DataFrame(data, var_names=clean_cols)
    
    cmi = CMIknn(significance='shuffle_test', knn=0.2, shuffle_neighbors=5, transform='ranks', workers=-1)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cmi, verbosity=0)
    results = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=0.05)
    
    discovered_edges = []
    p_matrix = results['p_matrix']
    graph = results['graph'] if 'graph' in results else None
    
    for i, var1 in enumerate(clean_cols):
        for j, var2 in enumerate(clean_cols):
            for tau in range(tau_max + 1):
                if i == j and tau == 0: continue
                
                # Check graph output if available since val_matrix / p_matrix is symmetric for tau=0
                if graph is not None:
                    # graph[i, j, 0] == '-->' or 'o-o' is a link.
                    # but tigramite graph shape is (N, N, tau_max+1)
                    val = graph[i, j, tau]
                    if val == '-->' or val == 'o-o' or val == 'x-x':
                        discovered_edges.append((f"{var1}_lag{tau}", f"{var2}_lag0"))
                else:
                    if p_matrix[i, j, tau] < 0.05:
                        discovered_edges.append((f"{var1}_lag{tau}", f"{var2}_lag0"))
                        
    return list(set(discovered_edges))

def calculate_metrics(discovered_edges, ground_truth_edges):
    true_positives = len(set(ground_truth_edges) & set(discovered_edges))
    false_positives = len(set(discovered_edges) - set(ground_truth_edges))
    false_negatives = len(set(ground_truth_edges) - set(discovered_edges))
    TPR = true_positives / len(ground_truth_edges) if ground_truth_edges else 0
    FDR = false_positives / len(discovered_edges) if discovered_edges else 0
    SHD = false_positives + false_negatives
    return TPR, FDR, SHD

def main():
    df = generate_synthetic_data(500)
    
    # Periods roughly spanning values near and far from actual frequencies
    # true frequencies in synthetic data: 12, 15, 30, 45
    # Let's test a baseline (correct-ish) and some wrong values.
    # What if no decomposition? period = 0
    periods = [0, 15, 30]
    lags_list = [2, 4, 6]
    
    print("Lag\tPeriod\tTPR\tFDR\tSHD", flush=True)
    print("-" * 45, flush=True)
    
    for lag in lags_list:
        for period in periods:
            df_resid = decompose_all_fixed(df, period)
            discovered_edges = df_to_pcmci(df_resid, tau_max=lag)
            tpr, fdr, shd = calculate_metrics(discovered_edges, ground_truth_edges)
            print(f"{lag}\t{period}\t{tpr:.3f}\t{fdr:.3f}\t{shd}", flush=True)

if __name__ == '__main__':
    main()
