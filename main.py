from clustering import *
from tools import *


def test_spectral_clustering(n_individuals=2):
    data, labels = load_Yale_data()
    if n_individuals<38:
        n_max = np.where(labels==n_individuals)[0][0]
    else:
        n_max = data.shape[1]
    # n_max is the first index of individual of id n_individual+1
    for sigma in [0.1, 0.3, 0.3, 0.5, 0.7, 0.9, 1]:
        affinity = compute_affinity_matrix(data, K=3, sigma=sigma, n_pictures=n_max)
        labels = labels[:n_max]
        # print("Starting spectral clustering")
        pred_labels = SpectralClustering(affinity, n=n_individuals)
        error = clustering_error(pred_labels, labels[:n_max])
        print("prediction error for %i individuals, sigma=%i : %.2f%%" %(n_individuals, sigma, 100*error))


def test_ksubspaces_clustering(n_individuals=2):
    data, labels = load_Yale_data()
    if n_individuals<38:
        n_max = np.where(labels==n_individuals)[0][0]
    else:
        n_max = data.shape[1]
    # n_max is the first index of individual of id n_individual+1
    for d in [1, 5, 10, 38, 50]:
        labels = labels[:n_max]
        data = data[:, :n_max]
        # print("Starting spectral clustering")
        pred_labels = ksubspaces(data, n_individuals, d, replicates=4)
        error = clustering_error(pred_labels, labels[:n_max])
        print("prediction error for %i individuals, d=%i : %.2f%%" %(n_individuals, d, 100*error))


if __name__ == "__main__":
    test_ksubspaces_clustering(n_individuals=3)
