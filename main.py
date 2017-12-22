from clustering import *
from tools import *
import matplotlib.pyplot as plt


def test_SC():
    print("#"*50)
    print("Tests on Spectral Clustering")
    print("#"*50)
    data_, labels_ = load_Yale_data()
    n_individuals=2
    n_max = np.where(labels_==n_individuals)[0][0]
    labels = labels_[:n_max]
    data = data_[:, :n_max]

    best_error=100
    best_params = {}
    perfs = {}
    sigmas = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7, 0.9, 1]
    Ks = [1, 2, 3, 4, 5, 6, 7, 8]
    for sigma in sigmas:
        for K in Ks:
            affinity = compute_affinity_matrix(data, K=K, sigma=sigma, n_pictures=n_max)
            pred_labels = SpectralClustering(affinity, n=n_individuals)
            error = clustering_error(pred_labels, labels)
            perfs[(K, sigma)] = error
            if error<best_error:
                best_error = error
                best_params['K']=K
                best_params['sigma'] = sigma
    print("Testing on individual 1-2 finished, best parameters are : K=%i, sigma=%.1f" %(best_params['K'], best_params['sigma']))

    to_plot_K_fixed = [perfs[(best_params['K'], sigma)] for sigma in sigmas]
    plt.plot(sigmas, to_plot_K_fixed)
    plt.title("Evolution of error with Sigma at optimal K (with individuals 1-2)")
    plt.ylabel("Error")
    plt.xlabel("sigma")
    plt.show()

    to_plot_sigma_fixed = [perfs[(K, best_params['sigma'])] for K in Ks]
    plt.plot(Ks, to_plot_sigma_fixed)
    plt.title("Evolution of error with K at optimal Sigma (with individuals 1-2)")
    plt.ylabel("Error")
    plt.xlabel("K")
    plt.show()

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


def test_ksubspaces_clustering():
    print("#"*50)
    print("Tests on K-subspaces clustering")
    print("#"*50)
    data_, labels_ = load_Yale_data()
    n_individuals=2
    n_max = np.where(labels_==n_individuals)[0][0]
    labels = labels_[:n_max]
    data = data_[:, :n_max]


    dimensions = [5,6,7,8,9,10,11,12]
    Ks = [8]
    best_error=100
    best_params = {}
    perfs = {}
    for d in dimensions:
        for K in Ks:
        # print("Starting spectral clustering")
            pred_labels = ksubspaces(data, n_individuals, d, replicates=K)
            error = clustering_error(pred_labels, labels)
            # print("prediction error for %i individuals, dimension of subspaces : %i, %i replicates : %.2f%%" %(n_individuals, d, K, 100*error))
            perfs[(K, d)] = error
            if error<best_error:
                best_error = error
                best_params['K']=K
                best_params['d'] = d

    print("Testing on individual 1-2 finished, the best parameter is, for %i replicates, dimension=%i" %(best_params['K'], best_params['d']))


    to_plot_K_fixed = [perfs[(best_params['K'], d)] for d in dimensions]
    plt.plot(dimensions, to_plot_K_fixed)
    plt.title("Evolution of error with dimension with 8 replicates (with individuals 1-2)")
    plt.ylabel("Error")
    plt.xlabel("dimension of subspaces")
    plt.show()


    K = best_params['K']
    d = best_params['d']
    for n_individuals in [2, 10, 20, 30, 38]:
        if n_individuals<38:
            n_max = np.where(labels_==n_individuals)[0][0]
        else:
            n_max = data_.shape[1]
        labels = labels_[:n_max]
        data = data_[:, :n_max]
        pred_labels = ksubspaces(data, n_individuals, d, replicates=K)
        error = clustering_error(labels, pred_labels)
        print("Error for %i individuals K-subspaces clustering : %.1f%% " %(n_individuals, 100*error))


def test_SSC():
    print("#"*50)
    print("Tests on Sparse Subspace Clustering")
    print("#"*50)
    data_, labels_ = load_Yale_data()
    n_individuals=2
    n_max = np.where(labels_==n_individuals)[0][0]
    labels = labels_[:n_max]
    data = data_[:, :n_max]


    taus = [0.1, 1, 5, 10, 25, 50, 100]
    mus = [1, 5, 10, 20]
    Ks = [8]
    best_error=100
    best_params = {}
    perfs = {}
    for tau in taus:
        for mu in mus:
            pred_labels = SSC(data, n_individuals, tau, mu)
            error = clustering_error(labels, pred_labels, verbose = False)
            # print("prediction error for tau= %i, mu=%i : %.2f%%" %(tau, mu, 100*error))
            perfs[(tau, mu)] = error
            if error<=best_error:
                best_error = error
                best_params['tau']=tau
                best_params['mu'] = mu
    print("Testing on individual 1-2 finished, the best parameters are, for tau= %.1f, mu=%i" %(best_params['tau'], best_params['mu']))

    to_plot_mu_fixed = [perfs[(tau, best_params['mu'])] for tau in taus]
    plt.plot(taus, to_plot_mu_fixed)
    plt.title("Evolution of error with lambda at optimal mu (with individuals 1-2)")
    plt.ylabel("Error")
    plt.xlabel("Tau")
    plt.show()


    tau = best_params['tau']
    mu = best_params['mu']
    for n_individuals in [2, 10, 20, 30, 38]:
        if n_individuals<38:
            n_max = np.where(labels_==n_individuals)[0][0]
        else:
            n_max = data_.shape[1]
        labels = labels_[:n_max]
        data = data_[:, :n_max]
        pred_labels = SSC(data[:,:n_max], n_individuals, tau, mu)
        error = clustering_error(pred_labels, labels)
        print("Error for %i individuals SSC : %.1f%% " %(n_individuals, 100*error))


if __name__ == "__main__":
    test_SC()
    test_ksubspaces_clustering()
    test_SSC()
