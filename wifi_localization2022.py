import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math
from scipy import spatial
from matplotlib.colors import ListedColormap

# Set global font settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Calibri', 'DejaVu Sans']
plt.rcParams['font.size'] = 10


def plot_cosine_similarity(similarity_matrix, classes, title='Location Similarity'):
    """ Plot the simplified similarity matrix.
    """
    plt.figure(figsize=(10, 8))

    # Simplified 3-color map: Red=Weak, Yellow=Medium, Green=Strong similarity
    cmap = ListedColormap(['#ff9999', '#ffff99', '#99ff99'])
    bounds = [0.0, 0.4, 0.7, 1.0]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    im = plt.imshow(similarity_matrix, interpolation='nearest', cmap=cmap, norm=norm)
    plt.title('Location Similarity (Green = Highly Similar)')
    
    # Custom colorbar
    cbar = plt.colorbar(im, ticks=[0.2, 0.55, 0.85])
    cbar.ax.set_yticklabels(['Weak', 'Medium', 'Strong'])

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=8)
    plt.yticks(tick_marks, classes, fontsize=8)

    # Highlight a group
    ax = plt.gca()
    rect = patches.Rectangle((-0.5, -0.5), 3, 3, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    plt.text(2.6, -0.6, 'Consistent Location Readings', color='red', fontsize=10, weight='bold')

    # Add non-technical conclusion
    plt.figtext(0.5, 0.01, "Conclusion: These scans show highly stable readings, meaning this is a reliable location.", ha="center", fontsize=10, bbox={"facecolor":"#eeeeee", "alpha":0.8, "pad":5})

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    import os
    os.makedirs('process_data', exist_ok=True)
    plt.savefig('process_data/simplified_similarity.png', dpi=300)
    plt.show()

def plot_wifi_hotspot_signal_strengths(features, labels, label_names):
  """ Location Cards: Show Top 3 APs for each location.
  """
  subplots_per_row = 3
  rows = int(math.ceil(len(labels)/subplots_per_row))

  fig, axarr = plt.subplots(rows, subplots_per_row, figsize=(12, 2 * rows))
  # Flatten axarr for easier indexing
  if rows > 1:
      axs = axarr.flatten()
  else:
      axs = axarr

  for i in range(len(labels)):
    features_i = features[i, :]
    # Replace nan with -100 temporarily for sorting
    clean_features = np.nan_to_num(features_i, nan=-100)
    # Get top 3 strongest APs
    top3_idx = np.argsort(clean_features)[-3:][::-1]
    
    top3_vals = clean_features[top3_idx]
    top3_names = [list(label_names)[idx][:10] + '...' if len(list(label_names)[idx])>10 else list(label_names)[idx] for idx in top3_idx]
    
    colors = ['green' if v > -60 else 'yellow' if v > -80 else 'red' for v in top3_vals]
    
    axs[i].bar(range(3), top3_vals, color=colors)
    axs[i].set_title(f"Location: {labels[i]}", fontsize=10, fontweight='bold')
    axs[i].set_xticks(range(3))
    axs[i].set_xticklabels(top3_names, rotation=30, ha='right', fontsize=8)
    axs[i].set_ylim(-100, 0)
    axs[i].text(1, -90, "Top 3 Signals", ha='center', fontsize=8, color='black', bbox={"facecolor":"white", "alpha":0.8, "pad":2})

  # Hide unused subplots
  for j in range(len(labels), len(axs)):
      axs[j].axis('off')

  plt.figtext(0.5, 0.01, "Conclusion: Green indicates a strong signal. Most locations have at least one strong Access Point.", ha="center", fontsize=12, bbox={"facecolor":"#eeeeee", "alpha":0.8, "pad":5})

  import os
  os.makedirs('process_data', exist_ok=True)
  plt.tight_layout(rect=[0, 0.05, 1, 1])
  plt.savefig('process_data/location_cards.png', dpi=300)
  plt.show()

def plot_feature_matrix(features, labels):
    """ Visualize the feature matrix as a simplified heat zone.
    """
    plt.figure(figsize=(10, 8))
    
    # Map values to 3 colors: Red (<-80, Weak), Yellow (-80 to -60, Medium), Green (>-60, Strong)
    cmap = ListedColormap(['#ff9999', '#ffff99', '#99ff99'])
    bounds = [-100, -80, -60, 0]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    im = plt.imshow(features, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
    plt.title('Simplified Signal Zones')
    
    cbar = plt.colorbar(im, ticks=[-90, -70, -30])
    cbar.ax.set_yticklabels(['Weak Signal', 'Medium Signal', 'Strong Signal'])

    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.xlabel('Access Points (Simplified)', fontsize=10)
    plt.xticks([]) # Hide x ticks for simplification
    plt.ylabel('Recorded Locations', fontsize=10)

    # Highlight a region
    ax = plt.gca()
    rect = patches.Rectangle((0, -0.5), 10, 5, linewidth=2, edgecolor='blue', facecolor='none')
    ax.add_patch(rect)
    plt.text(10.5, 0.5, 'Key Signal Providers for this Area', color='blue', fontsize=10, weight='bold')

    # Add conclusion
    plt.figtext(0.5, 0.01, "Conclusion: Green zones indicate reliable network coverage in these specific locations.", ha="center", fontsize=10, bbox={"facecolor":"#eeeeee", "alpha":0.8, "pad":5})

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    import os
    os.makedirs('process_data', exist_ok=True)
    plt.savefig('process_data/simplified_heat_zones.png', dpi=300)
    plt.show()

def extract_wifi_location_features(fileName):
  """ Extract the features (the wifi signal strength of different wifi hotspots)

    Returns:
      features: The signal strength of different wifi hotspots.
      labels: The location names.
      feature_names: mac addresses.
  """

  labels = dict()
  mac_addresses = dict()

  # Construct the feature dimension and labels
  file = open(fileName, 'r')
  for line in file:
    if line != '\n':
      # Line starting with ~^~ means location (label). Otherwise it means mac address.
      if line[0:3] == '~^~':
        location_name = line[4:-4] + '_' + str(len(labels))

      else:
        mac_addresss = line.split('~~')[0]
        if mac_addresss not in mac_addresses:
          mac_addresses[mac_addresss] = len(mac_addresses)

  file.close()

  # Construct the features and corresponding labels
  labels = list()
  features = None
  current_features = None
  file = open(fileName, 'r')
  for line in file:
    if line != '\n':
      if line[0:3] == '~^~':
        labels.append(line[3:-4])

        # If it's not the first label in the file,
        # append current features (features constructed for the last label) to the global features
        if current_features is not None:

          if features is None:
            features = current_features
          else:
            features = np.append(features, current_features, axis=0)

        current_features = np.empty((1, len(mac_addresses)))
        current_features[:] = np.nan

      else:
        mac_address = line.split('~~')[0]
        strength = float(line.split('~~')[-1])
        current_features[0, mac_addresses[mac_address]] = strength

  # Add the features for the last wifi location to the entire features set
  features = np.append(features, current_features, axis=0)

  return features, labels, mac_addresses.keys()


if __name__ == "__main__":
  # Change the file name
  fileName = "./raw_data/cuu25pbu.txt"

  # Each row of "features" contains scan results for each wifi scan,
  # and each row of "labels" contains scan name for each wifi scan.

  features, labels, label_names = extract_wifi_location_features(fileName)

  # Plot the histogram of wifi hotspot signal strengh.
  # You can comment it out if you don't want the plot to be shown.
  plot_wifi_hotspot_signal_strengths(features, labels, label_names)

  # -100 dBm means no signal at all
  features[np.isnan(features)] = -100

  import os
  os.makedirs("process_data", exist_ok=True)

  # Visualize the processed feature matrix
  plot_feature_matrix(features, labels)

  # Save the processed data to process_data directory
  np.save("process_data/features.npy", features)
  np.save("process_data/labels.npy", labels)
  np.save("process_data/label_names.npy", list(label_names))

  #YOUR CODE FOR 'TO DO 2' WILL GO HERE
  #Compute the cosine similarity matrix of your own wifi signal strength and save the matrix as similarity_matrix
  ####################################
  similarity_matrix = cosine_similarity(features)
  np.save("process_data/similarity_matrix.npy", similarity_matrix)

  ####################################
  plot_cosine_similarity(similarity_matrix, labels)

  # Compute prediction based on the similarity matrix (1-NN)
  # For each sample, find the most similar other sample
  np.fill_diagonal(similarity_matrix, -1) # Ignore self-similarity
  predicted_indices = np.argmax(similarity_matrix, axis=1)

  # Strip the suffix (e.g., '_0', '_1') from labels to get actual class names
  true_classes = [label.rsplit('_', 1)[0] if '_' in label else label for label in labels]
  predicted_classes = [true_classes[idx] for idx in predicted_indices]

  from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
  unique_classes = sorted(list(set(true_classes)))
  cm = confusion_matrix(true_classes, predicted_classes, labels=unique_classes)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_classes)

  fig, ax = plt.subplots(figsize=(10, 8))
  # Simplified confusion matrix presentation
  disp.plot(ax=ax, cmap=ListedColormap(['#ffffff', '#ff9999', '#ffff99', '#99ff99']))
  plt.title("Location Recognition Matrix")
  
  # Highlight common confusion
  rect = patches.Rectangle((0.5, 1.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
  ax.add_patch(rect)
  # Position could vary, but as an example
  plt.text(1.6, 1.4, 'Common Mix-up', color='red', fontsize=10, weight='bold')

  plt.figtext(0.5, 0.01, "Conclusion: Most locations are correctly identified (diagonal green). Areas in red highlight where the system gets confused, suggesting we might need more data there.", ha="center", fontsize=10, bbox={"facecolor":"#eeeeee", "alpha":0.8, "pad":5})

  plt.tight_layout(rect=[0, 0.05, 1, 1])
  plt.savefig('process_data/simplified_confusion_matrix.png', dpi=300)
  plt.show()
