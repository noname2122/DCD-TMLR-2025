from pathlib import Path

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from causalnex.structure.dynotears import from_pandas_dynamic

# Resolve repo paths relative to this file so the script works from any CWD.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "datasets" / "ehh1.csv"
FIG_PATH = REPO_ROOT / "figures" / "ehh1_dynotears_graph.png"
FIG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Load ehh1 dataset
df_ehh1 = pd.read_csv(DATA_PATH)

print("Running DYNOTEARS on ehh1.csv (p=2, lambda=0.1)...")
sm_ehh1 = from_pandas_dynamic(
    df_ehh1,
    p=2,                  
    lambda_w=0.1,        
    lambda_a=0.1,        
    w_threshold=0.1      
)

# Plot
graph_ehh1 = nx.DiGraph()
for source, targets in sm_ehh1.adj.items():
    for target, edge_data in targets.items():
        if abs(edge_data['weight']) > 0.1:  # threshold used in DYNOTEARS.py
            graph_ehh1.add_edge(source, target, weight=edge_data['weight'])

plt.figure(figsize=(15, 10))
pos = nx.spring_layout(graph_ehh1, k=2)
nx.draw(graph_ehh1, pos, with_labels=True, node_color='lightgreen',
        node_size=1000, font_size=8, font_weight='bold',
        edge_color='gray', arrows=True)

edge_labels = nx.get_edge_attributes(graph_ehh1, 'weight')
edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
nx.draw_networkx_edge_labels(graph_ehh1, pos, edge_labels=edge_labels, font_size=7)

plt.title("Dynamic Causal Graph (DynoTEARS) for Dataset ehh1")
plt.savefig(FIG_PATH)
print(f"Total significant relationships found: {len(graph_ehh1.edges())}")
print(f"Graph saved to {FIG_PATH}")
