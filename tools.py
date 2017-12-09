import numpy as np
import scipy.io as spio
import time
import os


def load_Yale_data():
    data = spio.loadmat("data/ExtendedYaleB.mat")
    labels = data['EYALEB_LABEL']
    labels = labels[0] - 1
    pictures = data['EYALEB_DATA']
    return pictures, labels


def compute_affinity_matrix(data, K, sigma, load_from_file=False):
    # data is D by N
    # K : number of closest neighbours
    # sigma parameter of the gaussian
    D, N = data.shape
    Affinity = np.zeros((N, N))
    distance_matrix = np.zeros((N, N))

    # Computes distance matrix
    # TODO vectorize this function for the sake of efficiency
    if os.path.exists("data/distance_matrix.npy"):
        distance_matrix = np.load("data/distance_matrix.npy")
    else:
        print("Computing distance matrix, can take up to 2 minutes...")
        t0 = time.time()
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sum((data[:, i] - data[:, j])**2)
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        t1 = time.time()
        print("Time to compute distance matrix : %.1f s" %(t1-t0))
        np.save("data/distance_matrix.npy", distance_matrix)

    # Computes affinity matrix
    if os.path.exists("data/affinity_matrix.npy") and load_from_file:
        Affinity = np.load("data/affinity_matrix.npy")
    else:
        for i in range(N):
            distances = distance_matrix[i, :]
            lowest = np.argsort(distances)[:(K+1)]
            for j in range(1, K+1):
                # Not taking the lowest distance which would be 0 = d(x, x)
                dist = distances[lowest[j]]
                Affinity[i, lowest[j]] += np.exp(-dist**2/(2*sigma**2))/2.
                Affinity[lowest[j], i] += np.exp(-dist**2/(2*sigma**2))/2.
                # Symmetrizes the affinity matrix
        np.save("data/affinity_matrix.npy", Affinity)
    return Affinity


def build_cost_matrix(true_labels, predicted_labels, nb_label):
    n = len(true_labels)
    cost_matrix = np.zeros((nb_label,nb_label))
    for i in range(nb_label):
        for j in range(nb_label):
            nb_error = 0
            for k in range(n):
                if predicted_labels[k] == j and true_labels[k] != i:
                    nb_error += 1
            cost_matrix[i,j] = nb_error
    return cost_matrix


if __name__=="__main__":
    pict, labels = load_Yale_data()
    compute_affinity_matrix(pict, 5, 5)
    pass
