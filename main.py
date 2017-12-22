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
    import matplotlib.pyplot as plt
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
        affinity = compute_affinity_matrix(data, K=K, sigma=sigma, n_pictures=n_max, dataset_name = "Yale")
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

def test_motion_segmentation():
    for dataset in ["cars2B", "1R2RC", "kanatani1"]:
        print("---- %s dataset"%(dataset))
        data, labels, n_individuals = load_Hopkins_data(dataset)
        print("Number of groups : %i"%(n_individuals))

        best_error=100
        best_params = {}
        perfs = {}
        sigmas = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7, 0.9, 1]
        Ks = [1, 2, 3, 4, 5, 6, 7, 8]
        for sigma in sigmas:
            for K in Ks:
                affinity = compute_affinity_matrix(data, K=K, sigma=sigma, n_pictures=len(data[0]), dataset_name = dataset)
                pred_labels = SpectralClustering(affinity, n=n_individuals)
                error = clustering_error(pred_labels, labels)
                perfs[(K, sigma)] = error
                if error<best_error:
                    best_error = error
                    best_params['K']=K
                    best_params['sigma'] = sigma
        print("Testing on dataset %s finished, best parameters are : K=%i, sigma=%.1f" %(dataset, best_params['K'], best_params['sigma']))
        import matplotlib.pyplot as plt
        to_plot_K_fixed = [perfs[(best_params['K'], sigma)] for sigma in sigmas]
        plt.plot(sigmas, to_plot_K_fixed)
        plt.title("Evolution of error with Sigma at optimal K for dataset %s"%(dataset))
        plt.ylabel("Error")
        plt.xlabel("sigma")
        plt.show()

        to_plot_sigma_fixed = [perfs[(K, best_params['sigma'])] for K in Ks]
        plt.plot(Ks, to_plot_sigma_fixed)
        plt.title("Evolution of error with K at optimal Sigma for dataset %s"%(dataset))
        plt.ylabel("Error")
        plt.xlabel("K")
        plt.show()

        perfs = {}
        best_error = 100
        taus = [2, 5, 10, 25, 100]
        mus = [1, 3, 5, 8, 10]
        for tau in taus:
            for mu in mus:
                pred_labels = SSC(data, n_individuals, tau, mu)
                error = clustering_error(pred_labels, labels)
                perfs[(tau, mu)] = error
                if error<best_error:
                    best_error = error
                    best_params['tau']=tau
                    best_params['mu'] = mu
        print("Testing on dataset %s finished, best parameters are : tau=%i, mu=%.1f" %(dataset, best_params['tau'], best_params['mu']))
        import matplotlib.pyplot as plt
        to_plot_tau_fixed = [perfs[(best_params['tau'], mu)] for mu in mus]
        plt.plot(mus, to_plot_tau_fixed)
        plt.title("Evolution of error with mu at optimal tau for dataset %s"%(dataset))
        plt.ylabel("Error")
        plt.xlabel("mu")
        plt.show()

        to_plot_mu_fixed = [perfs[(tau, best_params['mu'])] for tau in taus]
        plt.plot(taus, to_plot_mu_fixed)
        plt.title("Evolution of error with tau at optimal mu for dataset %s"%(dataset))
        plt.ylabel("Error")
        plt.xlabel("tau")
        plt.show()


if __name__ == "__main__":
    # test_ksubspaces_clustering(n_individuals=3)
    test_motion_segmentation()
