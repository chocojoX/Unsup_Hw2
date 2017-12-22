from tools import *
import numpy as np
from sklearn import cluster
from munkres import Munkres
import scipy
import time


def SpectralClustering(Affinity, n):
    # Affinity: N by N affinity matrix, where N is the number of points.
    # n: number of groups
    degrees = []
    for i in range(Affinity.shape[0]):
        degrees.append(np.sum(Affinity[i, :]))
    D = np.diag(degrees)
    L = D - Affinity

    eig_val, eig_vect = np.linalg.eig(L)
    eig_values_order = np.argsort(eig_val)[1:]

    # Get the n lowest eigen values and eigen vectors associated to them
    eig_val = eig_val[eig_values_order[:n]]
    Y = eig_vect[:, eig_values_order[:n]]

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

    D, N = data.shape
    """ Initialization """
    U_matrices = []
    U_Ut_matrices = []
    mu_vectors = []
    for i in range(n):
        U = get_random_orthogonal_matrix(D, d)
        U_matrices.append(U)
        U_Ut_matrices.append(np.dot(U, np.transpose(U)))
        mu = np.random.rand(D)
        mu_vectors.append(mu)

    """ Iterations """
    converged = False
    Y = np.zeros((d, N))
    #total_distance = 99999999999999999999999
    w_old = np.zeros((n, N))
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
                a = np.sum((np.dot(np.eye(D)-U_Ut, data-mu))**2,
                axis=0)
                a = a.reshape(1, len(a))
                distances = np.concatenate((distances,a),axis = 0)

        indexes = np.argmin(distances,axis=0)

        for j in range(N):
            w[indexes[j],j] = 1

        """ Find an estimation of the best subspaces given segmentation """
        for subspace_idx in range(n):
            t_mu=time.time()
            idx = np.where(w[subspace_idx, :]==1)[0]
            mu = np.zeros((data.shape[0]))
            for l in idx:
                mu += data[:, l]/len(idx)
            mu_vectors[subspace_idx] = mu

            covariance = np.dot(data[:, idx], np.transpose(data[:, idx]))
            U = SVD(covariance, d=d)
            U = U[:, :d]
            U_matrices[subspace_idx] = U
            U_Ut_matrices[subspace_idx] = np.dot(U, np.transpose(U))

        #print(total_distance)
        if np.sum(np.abs(w_old - w)) == 0:
            converged = True
        w_old = np.copy(w)

    pred_labels = np.argmax(w, axis=0)
    return pred_labels



def SSC(data, n, tau, mu2):
    # data: D by N data matrix.
    # n: number of clusters
    # tau, mu2: parameter
    #TODO
    return


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
    if verbose>1:
        print(dic)
    error = 0
    for k in range(len(label)):
        if label[k] != dic[groups[k]]:
            error += 1

    return error/len(label)


if __name__=="__main__":
    data, labels = load_Yale_data()
    # affinity = compute_affinity_matrix(data, K=5, sigma=200000)
    # print("Starting spectral clustering")
    # pred_labels = SpectralClustering(affinity, n=38)
    # error = clustering_error(pred_labels, labels)
    # print("prediction error : %.2f%%" %(100*error))

    pred_labels = ksubspaces(data[:,:128], 2, 3, 1)

    error = clustering_error(labels[:128], pred_labels, verbose = True)
    print("prediction error : %.2f%%" %(100*error))
    print(labels[:128])
    print(pred_labels)


    pass
