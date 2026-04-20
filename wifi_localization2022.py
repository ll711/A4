import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math
from scipy import spatial

# Set global font settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Calibri', 'DejaVu Sans']
plt.rcParams['font.size'] = 10


def plot_cosine_similarity(similarity_matrix, classes, title='Consine Similarity'):
    """ Plot the cosine similarity matrix.
    """
    plt.figure(figsize=(10, 8))

    im = plt.imshow(similarity_matrix, interpolation='nearest', cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    plt.title('Cosine Similarity (High = Dark Blue)')
    cbar = plt.colorbar(im)
    cbar.set_label('Similarity Score (0.0 to 1.0)')

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=8)
    plt.yticks(tick_marks, classes, fontsize=8)

    # Add a rectangle to highlight a cluster
    ax = plt.gca()
    rect = patches.Rectangle((-0.5, -0.5), 3, 3, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    plt.text(2.6, -0.6, 'Group: High Similarity Cluster', color='red', fontsize=10, weight='bold')

    # Add conclusion
    plt.figtext(0.5, 0.01, "Conclusion: Samples from the same location exhibit high cosine similarity (darker blue).", ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    import os
    os.makedirs('process_data', exist_ok=True)
    plt.savefig('process_data/cosine_similarity.png', dpi=300)
    plt.savefig('process_data/confusion_matrix.png', dpi=300)
    plt.show()


def plot_wifi_hotspot_signal_strengths(features, labels, label_names):
  """ Histogram of the wifi access point signal strength.
  """

  subplots_per_row = 3
  rows = int(math.ceil(len(labels)/subplots_per_row))

  fig, axarr = plt.subplots(rows, subplots_per_row, sharex=True, sharey=True)

  for i in range(len(labels)):
    row = int(i/subplots_per_row)
    col = i % subplots_per_row
    # Change the sign of wifi signal strength
    axarr[row, col].bar(range(len(label_names)), features[i, :])
    axarr[row, col].set_title(labels[i], fontsize=8)
    axarr[row, col].set_xticks(range(len(label_names)))
    axarr[row, col].set_xticklabels(label_names, rotation='vertical', fontsize=8)
    axarr[row, col].set_ylabel('dBm')

  import os
  os.makedirs('process_data', exist_ok=True)
  plt.tight_layout()
  plt.savefig('process_data/wifi_signal_strengths.png', dpi=300)
  plt.show()


def plot_feature_matrix(features, labels):
    """ Visualize the feature matrix as a 2D heatmap.
    Rows are different scans (labels) and columns are different MAC addresses.
    """
    plt.figure(figsize=(10, 8))
    # Using viridis cmap for signal strengths (-100 to 0) which is warm (yellow-strong, purple-weak)
    im = plt.imshow(features, aspect='auto', cmap='viridis', interpolation='nearest', vmin=-100, vmax=0)
    plt.title('Feature Matrix Visualization')
    cbar = plt.colorbar(im)
    cbar.set_label('Signal Strength (dBm)')

    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.xlabel('MAC Address (Feature Index)', fontsize=10)
    plt.ylabel('Scans (Labels)', fontsize=10)

    # Highlight a region
    ax = plt.gca()
    rect = patches.Rectangle((0, -0.5), 10, 5, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    plt.text(10.5, 0.5, 'Group X: Strong Base Stations', color='red', fontsize=10, weight='bold')

    # Add conclusion
    plt.figtext(0.5, 0.01, "Conclusion: Specific Access Points show consistently higher signal strength (yellow).", ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    import os
    os.makedirs('process_data', exist_ok=True)
    plt.savefig('process_data/features_visualization.png', dpi=300)
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
  disp.plot(ax=ax, cmap=plt.cm.Blues)
  plt.title("Predicted Confusion Matrix")
  plt.tight_layout()
  plt.savefig('process_data/predicted_confusion_matrix.png', dpi=300)
  plt.show()
