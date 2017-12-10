from tools import *
import numpy as np
from sklearn import cluster
from munkres import Munkres


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


def ksubspaces(data, n, d, replicates):
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
    total_distance = 99999999999999999999999
    while not converged:
        old_total_distance = total_distance
        total_distance = 0

        """ Find best subspace for each data point """
        w = np.zeros((n, N))
        for j in range(N):
            x = data[:, j]
            min_dist = 999999999999999999
            best_subspace = -1
            for subspace_idx in range(n):
                # U = U_matrices[subspace_idx]
                mu = mu_vectors[subspace_idx]
                U_Ut = U_Ut_matrices[subspace_idx]
                dist = np.sum( (np.dot(np.eye(D)-U_Ut, x-mu))**2 )
                if dist<min_dist:
                    min_dist=dist
                    best_subspace = subspace_idx
            w[best_subspace, j] = 1
            total_distance += min_dist/N

        """ Find an estimation of the best subspaces given segmentation """
        for subspace_idx in range(n):
            idx = np.where(w[subspace_idx, :]==1)[0]
            mu = np.zeros((data.shape[0]))
            for l in idx:
                mu += data[:, l]/len(idx)
            mu_vectors[subspace_idx] = mu

            covariance = np.sum(np.dot(data[:, l].reshape(-1, 1), data[:, l].reshape(1, -1)) for l in idx)
            U, S, V = SVD(covariance)
            U = U[:, :d]
            U_matrices[subspace_idx] = U
            U_Ut_matrices[subspace_idx] = np.dot(U, np.transpose(U))

        print(total_distance)
        if np.abs(old_total_distance- total_distance)<1:
            converged = True

    pred_labels = []
    for j in range(N):
        pred_labels.append(np.where(w[:, j]==1)[0][0])
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
    # TODO
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
    error = clustering_error(pred_labels, labels)
    print("prediction error : %.2f%%" %(100*error))



    pass
