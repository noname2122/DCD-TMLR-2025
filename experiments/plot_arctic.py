from pathlib import Path

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from causalnex.structure.dynotears import from_pandas_dynamic

# Resolve repo paths relative to this file so the script works from any CWD.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "datasets" / "Arctic_Monthly.csv"
FIG_PATH = REPO_ROOT / "figures" / "arctic_dynotears_graph.png"
FIG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Load Arctic Monthly dataset
df_arctic = pd.read_csv(DATA_PATH)

# Run DynoTEARS with multiple parameters to see if it changes
print("Running DYNOTEARS on Arctic_Monthly.csv (p=2, lambda=0.01)...")
sm_arctic = from_pandas_dynamic(
    df_arctic,
    p=2,                  
    lambda_w=0.01,        
    lambda_a=0.01,        
    w_threshold=0.05      
)

# Plot
graph_arctic = nx.DiGraph()
for source, targets in sm_arctic.adj.items():
    for target, edge_data in targets.items():
        if abs(edge_data['weight']) > 0.05:
            graph_arctic.add_edge(source, target, weight=edge_data['weight'])

plt.figure(figsize=(15, 10))
pos = nx.spring_layout(graph_arctic, k=3)
nx.draw(graph_arctic, pos, with_labels=True, node_color='lightblue',
        node_size=1000, font_size=8, font_weight='bold',
        edge_color='gray', arrows=True)

edge_labels = nx.get_edge_attributes(graph_arctic, 'weight')
edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
nx.draw_networkx_edge_labels(graph_arctic, pos, edge_labels=edge_labels, font_size=7)

plt.title("Dynamic Causal Graph (DynoTEARS) for Arctic Monthly Data")
plt.savefig(FIG_PATH)
print(f"Graph saved to {FIG_PATH}")
