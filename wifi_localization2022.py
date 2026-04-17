import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math
from scipy import spatial


def plot_cosine_similarity(similarity_matrix, classes, title='Consine Similarity'):
    """ Plot the cosine similarity matrix.
    """

    plt.imshow(similarity_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Cosine Similarity')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    import os
    os.makedirs('process_data', exist_ok=True)
    plt.savefig('process_data/cosine_similarity.png')
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
  plt.savefig('process_data/wifi_signal_strengths.png')
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
  import os
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

  # Save the processed data to process_data directory
  os.makedirs("process_data", exist_ok=True)
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
