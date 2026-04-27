import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def plot_cosine_similarity(similarity_matrix, classes, title='Cosine Similarity'):
    """ Plot the cosine similarity matrix.
    """

    plt.figure(figsize=(12, 12))
    plt.imshow(similarity_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Cosine Similarity Data Combined')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=5)
    plt.yticks(tick_marks, classes, fontsize=5)

    plt.tight_layout()
    if not os.path.exists('./process_data'):
        os.makedirs('./process_data')
    plt.savefig('./process_data/Similarity Data Combined.png', dpi=300)
    plt.close()

def plot_wifi_hotspot_signal_strengths(features, labels, label_names):
  """ Histogram of the wifi access point signal strength.
  """

  subplots_per_row = 3
  rows = int(np.ceil(len(labels)/subplots_per_row))

  fig, axarr = plt.subplots(rows, subplots_per_row, sharex=True, sharey=True)

  if rows > 1:
      axarr = axarr.flatten()
  else:
      axarr = np.array(axarr).flatten()

  for i in range(len(labels)):
    # Change the sign of wifi signal strength
    axarr[i].bar(range(len(label_names)), features[i, :])
    axarr[i].set_title(labels[i], fontsize=8)
    axarr[i].set_xticks(range(len(label_names)))
    axarr[i].set_xticklabels(label_names, rotation='vertical', fontsize=8)
    axarr[i].set_ylabel('dBm')

  # Hide any unused subplots
  for j in range(len(labels), len(axarr)):
      axarr[j].set_visible(False)

  plt.tight_layout()
  if not os.path.exists('./process_data'):
      os.makedirs('./process_data')
  plt.savefig('./process_data/wifi_hotspot_signal_strengths.png')
  plt.close()

def extract_wifi_location_features(fileName):
  """ Extract the features (the wifi signal strength of different wifi hotspots)

    Returns:
      features: The signal strength of different wifi hotspots.
      labels: The location names.
      feature_names: mac addresses.
  """

  mac_addresses = dict()

  # Construct the feature dimension and labels
  with open(fileName, 'r', encoding='utf-8') as file:
    for line in file:
      line = line.strip()
      if line:
        # Line starting with ~^~ means location (label). Otherwise it means mac address.
        if not line.startswith('~^~'):
          mac_address = line.split('~~')[0]
          if mac_address not in mac_addresses:
            mac_addresses[mac_address] = len(mac_addresses)

  # Construct the features and corresponding labels
  labels = list()
  features_list = list()
  current_features = None

  with open(fileName, 'r', encoding='utf-8') as file:
    for line in file:
      line = line.strip()
      if line:
        if line.startswith('~^~'):
          labels.append(line[3:-3] if line.endswith('~^~') else line[3:])

          # If it's not the first label in the file,
          # append current features (features constructed for the last label) to the global features
          if current_features is not None:
            features_list.append(current_features)

          current_features = np.full(len(mac_addresses), np.nan)

        else:
          parts = line.split('~~')
          mac_address = parts[0]
          strength = float(parts[-1])
          current_features[mac_addresses[mac_address]] = strength

  # Add the features for the last wifi location to the entire features set
  if current_features is not None:
    features_list.append(current_features)

  features = np.array(features_list)

  return features, labels, list(mac_addresses.keys())

if __name__ == "__main__":
  # Change the file name
  fileName = "./raw_data/both.txt"

  # Each row of "features" contains scan results for each wifi scan,
  # and each row of "labels" contains scan name for each wifi scan.

  features, labels, label_names = extract_wifi_location_features(fileName)

  # Plot the histogram of wifi hotspot signal strengh.
  # You can comment it out if you don't want the plot to be shown.
  # plot_wifi_hotspot_signal_strengths(features, labels, label_names)

  # -100 dBm means no signal at all
  features[np.isnan(features)] = -100

  #YOUR CODE FOR 'TO DO 2' WILL GO HERE  
  #Compute the cosine similarity matrix of your own wifi signal strength and save the matrix as similarity_matrix
  ####################################
  
  similarity_matrix = cosine_similarity(features)

  ####################################
  plot_cosine_similarity(similarity_matrix, labels)
