import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
import os
import wifi_localization2022
if not os.path.exists('./visual'):
    os.makedirs('./visual')
features, labels, _ = wifi_localization2022.extract_wifi_location_features('./raw_data/cuu25pbu.txt')
features[np.isnan(features)] = -100
# Calculate similarity and distance matrix
similarity_matrix = cosine_similarity(features)
distance_matrix = 1 - similarity_matrix
# MDS to 2D
mds = MDS(n_components=2, metric='precomputed', random_state=42, n_init=4, init='random')
coords = mds.fit_transform(distance_matrix)
# Group by location
loc_to_coords = {}
for i, label in enumerate(labels):
    base_loc = label.split('-')[0]
    if base_loc not in loc_to_coords:
        loc_to_coords[base_loc] = []
    loc_to_coords[base_loc].append(coords[i])
plt.figure(figsize=(10, 8))
# Extremely simplified floor plan background concept: just a light grid or bounding box
# plt.grid(True, linestyle='--', alpha=0.3)
plt.title('WiFi Fingerprint Constellation', fontsize=16)
colors = plt.cm.tab10(np.linspace(0, 1, len(loc_to_coords)))
for (loc, pts), color in zip(loc_to_coords.items(), colors):
    pts = np.array(pts)
    centroid = pts.mean(axis=0)
    # Plot centroid (star)
    plt.scatter(centroid[0], centroid[1], c=[color], marker='*', s=300, edgecolors='k', zorder=3)
    # Plot individual samples (small dots)
    plt.scatter(pts[:, 0], pts[:, 1], c=[color], marker='o', s=50, alpha=0.6, zorder=2)
    # Connect samples to centroid
    for p in pts:
        plt.plot([centroid[0], p[0]], [centroid[1], p[1]], c=color, linestyle=':', alpha=0.5)
    # Draw a halo (circle) around the centroid containing all points
    if len(pts) > 1:
        radius = np.max(np.linalg.norm(pts - centroid, axis=1)) 
        circle = plt.Circle(centroid, radius * 1.2, color=color, alpha=0.1, zorder=1)
        plt.gca().add_patch(circle)
    # Add label next to centroid
    plt.annotate(loc, (centroid[0], centroid[1]), xytext=(8, 8), textcoords='offset points', fontsize=12, fontweight='bold')
plt.axis('equal')
plt.axis('off') # Lo-Fi feel
plt.tight_layout()
plt.savefig('./visual/wifi_constellation.png', dpi=300, bbox_inches='tight')
print('Visualization saved to ./visual/wifi_constellation.png')

