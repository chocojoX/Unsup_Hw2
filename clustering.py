from tools import *
import numpy as np
from sklearn import cluster
from munkres import Munkres
import scipy
import time


def SpectralClustering(Affinity, n):
    # Affinity: N by N affinity matrix, where N is the number of points.
    # n: number of groups

    D = np.diag(np.sum(Affinity,axis=1))
    L = D - Affinity

    eig_val, eig_vect = np.linalg.eig(L)
    eig_values_order = np.argsort(eig_val)[1:]

    # Get the n lowest eigen values and eigen vectors associated to them
    eig_val = eig_val[eig_values_order[:n]]

    Y = eig_vect[:, eig_values_order[:n]]
    Y = Y / np.linalg.norm(Y, axis=0).reshape(1,-1)
    # Initialize K-means
    kmeans = cluster.KMeans(n_clusters = n)
    kmeans.fit(Y)
    predicted_labels = kmeans.predict(Y)
    return predicted_labels


def ksubspaces(data, n, d, replicates=1):
    # data: D by N data matrix.
    # n: number of subspaces
    # d: dimension of subspaces
    # replicates: number of restart
    precision = 0.01
    all_distances = []
    all_labels = []
    D, N = data.shape
    data = data/(np.linalg.norm(data, axis=0).reshape(1,-1))

    for rep in range(replicates):
        """ Initialization """
        U_matrices = []
        U_Ut_matrices = []
        mu_vectors = []
        for i in range(n):
            U = 0.1*get_random_orthogonal_matrix(D, d)
            U_matrices.append(U)
            U_Ut_matrices.append(np.dot(U, np.transpose(U)))
            mu = 0.*np.random.rand(D)
            mu_vectors.append(mu)

        """ Iterations """
        converged = False
        Y = np.zeros((d, N))
        w_old = np.zeros((n, N))
        distance_old = 1e5
        n_iter = 0
        while not converged and n_iter < 10:
            n_iter += 1

            """ Find best subspace for each data point """
            w = np.zeros((n, N))
            first = True
            for subspace_idx in range(n):
                mu = mu_vectors[subspace_idx].reshape(D,1)
                U_Ut = U_Ut_matrices[subspace_idx]
                if first:
                    distances = np.sum((np.dot(np.eye(D)-U_Ut, data-mu))**2, axis=0)
                    distances = distances.reshape(1,len(distances))
                    first = False
                else:
                    a = np.sum((np.dot(np.eye(D)-U_Ut, data-mu))**2, axis=0)
                    a = a.reshape(1, len(a))
                    distances = np.concatenate((distances,a),axis = 0)

            indexes = np.argmin(distances,axis=0)
            #total_distance = np.sum([distances[indexes[k], k] for k in range(len(indexes))])
            total_distance = np.sum(np.min(distances, axis=0))

            for j in range(N):
                w[indexes[j],j] = 1

            """ Find an estimation of the best subspaces given segmentation """
            mu_vectors = []
            U_matrices = []
            U_Ut_matrices = []
            for subspace_idx in range(n):
                t_mu=time.time()
                idx = np.where(w[subspace_idx, :]==1)[0]
                mu = np.zeros((data.shape[0]))
                for l in idx:
                    mu += data[:, l]/len(idx)
                mu_vectors.append(mu)

                covariance = np.dot(data[:, idx], np.transpose(data[:, idx]))
                U = partial_SVD(covariance, d=d)
                U = U[:, :d]
                U_matrices.append(U)
                U_Ut_matrices.append(np.dot(U, U.T))

            #print(total_distance)
            if n_iter > 2:
                #assert(distance_old - total_distance >=0)
                if distance_old - total_distance < precision:
                    converged = True
            distance_old = np.copy(total_distance)
            w_old = np.copy(w)

        pred_labels = np.argmax(w, axis=0)
        all_distances.append(total_distance)
        all_labels.append(pred_labels)

    # Return result which obtained the best objective score
    best_replicate = np.argmin(all_distances)
    return all_labels[best_replicate]



def SSC(data, n, tau, mu2):
    # data: D by N data matrix.
    # n: number of clusters
    # tau, mu2: parameter
    C = Lasso_minimization(data, mu2, tau)
    W = np.absolute(C) + np.absolute(C.T)
    predicted_labels = SpectralClustering(W, n)
    return predicted_labels


def clustering_error(label, groups, verbose=0):
    # label: N-dimensional vector with ground truth labels for a dataset with N points
    # groups: N-dimensional vector with estimated labels for a dataset with N points

    assert len(label) == len(groups)

    nb_label = len(np.unique(label))
    cost_matrix = build_cost_matrix(label, groups, nb_label)

    m = Munkres()
    indexes = m.compute(cost_matrix)

    dic = {}
    for i,j in indexes:
        dic[j] = i
    if verbose>0:
        print(dic)
    error = 0
    for k in range(len(label)):
        if label[k] != dic[groups[k]]:
            error += 1

    return error/len(label)


if __name__=="__main__":
    data, labels, n_individuals = load_Hopkins_data("kanatani1")

    #affinity = compute_affinity_matrix(data, K=5, sigma=2e6)

    #pred_labels = SpectralClustering(affinity, n=38)
    #error = clustering_error(pred_labels, labels, verbose=1)


    #pred_labels = ksubspaces(data[:,:], 2, 3, 1)
    print("Number of groups : %i"%(n_individuals))
    for tau in [10]:
        for mu in range(3,4):
            pred_labels = SSC(data, n_individuals, tau, mu)
            error = clustering_error(labels, pred_labels, verbose = False)
            print("prediction error for tau= %i, mu=%i : %.2f%%" %(tau, mu, 100*error))

    #
    # for K in range(2,6):
    #     for sigma in [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
    #         affinity = compute_affinity_matrix(data[:,:n_max], K=K, sigma=sigma)
    #         pred_labels = SpectralClustering(affinity, n=n_individuals)
    #         error = clustering_error(pred_labels, labels[:n_max], verbose=1)
    #         print("K={}, sigma={} : prediction error : {}".format(K,sigma,error))




    pass
