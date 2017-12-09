from tools import *
import numpy as np

def SpectralClustering(Affinity, n):
    # Affinity: N by N affinity matrix, where N is the number of points.
    # n: number of groups
    # TODO
    return


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
