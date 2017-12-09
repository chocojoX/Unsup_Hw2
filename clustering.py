from tools import *
import numpy as np
from sklearn.cluster import KMeans
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
    eig_values_order = np.argsort(eig_val)

    # Get the n lowest eigen values and eigen vectors associated to them
    eig_val = eig_val[eig_values_order[:n]]
    Y = eig_vect[:, eig_values_order[:n]]

    # Initialize K-means
    kmeans = KMeans(n_clusters = n)
    kmeans.fit(Y)
    predicted_labels = kmeans.predict(Y)
    return predicted_labels


def ksubspaces(data, n, d, replicates):
    # data: D by N data matrix.
    # n: number of subspaces
    # d: dimension of subspaces
    # replicates: number of restart
    #TODO
    return


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
    affinity = compute_affinity_matrix(data, K=5, sigma=200000)
    print("Starting spectral clustering")
    pred_labels = SpectralClustering(affinity, n=38)
    error = clustering_error(pred_labels, labels)
    print("prediction error : %.2f%%" %(100*error))
    pass
