from pathlib import Path

import pandas as pd
from causalnex.structure.dynotears import from_pandas_dynamic

# Resolve repo paths relative to this file so the script works from any CWD.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "datasets" / "Arctic_Monthly.csv"

# Load Arctic Monthly dataset
df_arctic = pd.read_csv(DATA_PATH)

print("Evaluating DYNOTEARS on Arctic_Monthly.csv...")
print(f"Shape: {df_arctic.shape}")
print(f"Columns: {df_arctic.columns.tolist()}")
print("-" * 50)

# Run DynoTEARS with same parameters as in the original script
try:
    sm_arctic = from_pandas_dynamic(
        df_arctic,
        p=2,                  
        lambda_w=0.01,        
        lambda_a=0.01,        
        w_threshold=0.05      
    )
    
    significant_edges = []
    for source, targets in sm_arctic.adj.items():
        for target, edge_data in targets.items():
            if abs(edge_data['weight']) > 0.05:
                significant_edges.append((source, target, edge_data['weight']))
                
    print(f"Total significant relational edges found (|weight| > 0.05): {len(significant_edges)}")
    for source, target, weight in sorted(significant_edges, key=lambda x: abs(x[2]), reverse=True):
        print(f"{source} --> {target} (Weight: {weight:.3f})")
except Exception as e:
    print(f"Failed to run DYNOTEARS: {e}")
