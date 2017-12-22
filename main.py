from clustering import *
from tools import *


def test_SC():
    data_, labels_ = load_Yale_data()
    n_individuals=2
    n_max = np.where(labels_==n_individuals)[0][0]
    labels = labels_[:n_max]
    data = data_[:, :n_max]

    best_error=100
    best_params = {}
    for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1]:
        for K in [3, 5, 7, 10, 15]:
            affinity = compute_affinity_matrix(data, K=K, sigma=sigma, n_pictures=n_max)
            pred_labels = SpectralClustering(affinity, n=n_individuals)
            error = clustering_error(pred_labels, labels)
            if error<best_error:
                best_error = error
                best_params['K']=K
                best_params['sigma'] = sigma
    print("Testing on individual 1-2 finished, best parameters are : K=%i, sigma=%.1f" %(best_params['K'], best_params['sigma']))
    K = best_params['K']
    sigma = best_params['sigma']
    for n_individuals in [2, 10, 20, 30, 38]:
        if n_individuals<38:
            n_max = np.where(labels_==n_individuals)[0][0]
        else:
            n_max = data_.shape[1]
        labels = labels_[:n_max]
        data = data_[:, :n_max]
        affinity = compute_affinity_matrix(data, K=K, sigma=sigma, n_pictures=n_max)
        pred_labels = SpectralClustering(affinity, n=n_individuals)
        error = clustering_error(pred_labels, labels)
        print("Error for %i individuals spectral clustering : %.1f%% " %(n_individuals, 100*error))


def test_ksubspaces_clustering(n_individuals=2):
    data, labels = load_Yale_data()
    if n_individuals<38:
        n_max = np.where(labels==n_individuals)[0][0]
    else:
        n_max = data.shape[1]
    # n_max is the first index of individual of id n_individual+1
    for d in [6,7,8,9,10]:
        labels = labels[:n_max]
        data = data[:, :n_max]
        # print("Starting spectral clustering")
        pred_labels = ksubspaces(data, n_individuals, d, replicates=5)
        error = clustering_error(pred_labels, labels)
        print("prediction error for %i individuals, d=%i : %.2f%%" %(n_individuals, d, 100*error))


if __name__ == "__main__":
    # test_ksubspaces_clustering(n_individuals=3)
    test_SC()
