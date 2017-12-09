from tools import *
import numpy as np
from sklearn.cluster import KMeans

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
    kmeans = Kmeans(n_cluster = n)
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


def clustering_error(label, groups):
    # label: N-dimensional vector with ground truth labels for a dataset with N points
    # groups: N-dimensional vector with estimated labels for a dataset with N points
    # TODO
    return


if __name__=="__main__":
    data, labels = load_Yale_data()
    print(data.shape)
    pass
