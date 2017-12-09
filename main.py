from clustering import *
from tools import *


def test_spectral_clustering(n_individuals=10):
    # import pdb; pdb.set_trace()
    data, labels = load_Yale_data()
    n_max = np.where(labels==n_individuals)[0][0]
    # n_max is the first index of individual of id n_individual+1
    affinity = compute_affinity_matrix(data, K=5, sigma=50000)
    affinity = affinity[:n_max, :n_max]
    labels = labels[:n_max]

    print("Starting spectral clustering")
    pred_labels = SpectralClustering(affinity, n=n_individuals)
    error = clustering_error(pred_labels, labels)
    print("prediction error for %i individuals : %.2f%%" %(n_individuals, 100*error))


if __name__ == "__main__":
    test_spectral_clustering(n_individuals=20)
