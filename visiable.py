import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ============================================
# 1. Prepare similarity matrix (Load from process_data)
# ============================================

# Read actual data
try:
    sim = np.load('process_data/similarity_matrix.npy')
    raw_labels = np.load('process_data/labels.npy').tolist()
    # Make names unique for the graph nodes
    locations = [f"{label}_{i}" for i, label in enumerate(raw_labels)]
except FileNotFoundError:
    print("Error: Could not find data files in process_data folder.")
    exit()

# Symmetrize if not perfectly symmetric
sim = (sim + sim.T) / 2
df_sim = pd.DataFrame(sim, index=locations, columns=locations)

# ============================================
# 2. Build network graph based on similarity (keep strong connections)
# ============================================
G = nx.Graph()
for loc in locations:
    G.add_node(loc)

threshold_strong = 0.75   # Similarity > 0.75: thick green line
threshold_medium = 0.5    # Similarity > 0.5: thin blue line
# threshold_weak = 0.3    # Similarity > 0.3: gray dotted line (optional)

for i, loc1 in enumerate(locations):
    for j, loc2 in enumerate(locations):
        if i < j:   # Upper triangular only to avoid duplicates
            sim_val = df_sim.loc[loc1, loc2]
            if sim_val >= threshold_strong:
                G.add_edge(loc1, loc2, weight=sim_val, style='strong')
            elif sim_val >= threshold_medium:
                G.add_edge(loc1, loc2, weight=sim_val, style='medium')

# ============================================
# 3. Draw graph
# ============================================
plt.figure(figsize=(14, 10))

# Use spring_layout
pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=1200, node_color='lightblue',
                       edgecolors='navy', linewidths=1.5)

# Draw labels (use original labes without suffix)
node_labels = {loc: loc.rsplit('_', 1)[0] for loc in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, font_weight='bold')

# Draw edges
strong_edges = [(u, v) for u, v, d in G.edges(data=True) if d['style'] == 'strong']
medium_edges = [(u, v) for u, v, d in G.edges(data=True) if d['style'] == 'medium']

# Strong similarity edges: Green thick lines
nx.draw_networkx_edges(G, pos, edgelist=strong_edges, width=3.5,
                       edge_color='forestgreen', alpha=0.9)
# Medium similarity edges: Blue thin dashed lines
nx.draw_networkx_edges(G, pos, edgelist=medium_edges, width=1.5,
                       edge_color='royalblue', alpha=0.6, style='dashed')

# Add legend
legend_elements = [
    Rectangle((0, 0), 1, 1, fc="forestgreen", alpha=0.9, label="Strongly Similar (≥0.75)"),
    Rectangle((0, 0), 1, 1, fc="royalblue", alpha=0.6, label="Moderately Similar (0.5–0.75)"),
]
plt.legend(handles=legend_elements, loc='upper left', fontsize=11)

plt.title("WiFi Signal Similarity Between Indoor Locations\n(Stage 3 Design: Intuitive Network View)",
          fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()

# Save image
os.makedirs('visual', exist_ok=True)
output_path = os.path.join('visual', 'stage3_similarity_network.png')
plt.savefig(output_path, dpi=200, bbox_inches='tight')
plt.show()

print(f"✅ Network graph saved to '{output_path}'")
print("💡 Explanation: Thick green lines connect locations with highly similar signal features.")
print("   Blue dashed lines indicate partial similarity.")
print("   Locations without connections have significantly different signal features.")
