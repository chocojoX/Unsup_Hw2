import numpy as np
import scipy.io as spio
import time
import os
from scipy.spatial.distance import cdist
from sklearn import decomposition

def load_Yale_data():
    data = spio.loadmat("data/ExtendedYaleB.mat")
    labels = data['EYALEB_LABEL']
    labels = labels[0] - 1
    pictures = data['EYALEB_DATA']
    return pictures, labels


def SVD(X):
    # Return, U, Sigma, V such that X = U.Sigma.V^T
    U, Sigma, V = np.linalg.svd(X, full_matrices=False)
    # Careful, np.linalg.svd return U, sigma, transpose(V) --> V need to be transposed.
    return U, np.diag(Sigma), np.transpose(V)

def partial_SVD(X, d=2000):
    # Return truncated U only
    svd = decomposition.TruncatedSVD(n_components=d)
    svd.fit(X)
    U = np.transpose(svd.components_)
    return U


def compute_affinity_matrix(data, K, sigma, load_from_file=False, verbose=0):
    # data is D by N
    # K : number of closest neighbours
    # sigma parameter of the gaussian
    D, N = data.shape
    n_pictures = N
    data_normalized = data/(np.linalg.norm(data, axis=0).reshape(1,-1))

    Affinity = np.zeros((n_pictures, n_pictures))
    distance_matrix = np.zeros((N, N))

    # Computes distance matrix

    if os.path.exists("data/distance_matrix.npy") and load_from_file:
        distance_matrix = np.load("data/distance_matrix.npy")
    else:
        if verbose>0:
            print("Computing distance matrix")

        t0 = time.time()
        """
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sum((data[:, i] - data[:, j])**2)
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        """

        distance_matrix = cdist(data_normalized.T,data_normalized.T, 'sqeuclidean')
        t1 = time.time()
        if verbose>0:
            print("Time to compute distance matrix : %.1f s" %(t1-t0))
        #np.save("data/distance_matrix.npy", distance_matrix)

    # Computes affinity matrix
    if os.path.exists("data/affinity_matrix.npy") and load_from_file:
        Affinity = np.load("data/affinity_matrix.npy")
    else:
        distance_matrix = distance_matrix[:n_pictures, :n_pictures]
        for i in range(n_pictures):
            #argsort
            distances = distance_matrix[i, :]
            lowest = np.argsort(distances)[:(K+1)]
            for j in range(1, K+1):
                # Not taking the lowest distance which would be 0 = d(x, x)
                dist = distances[lowest[j]]
                Affinity[i, lowest[j]] += np.exp(-dist/(2*sigma**2))
                Affinity[lowest[j], i] += np.exp(-dist/(2*sigma**2))
                # Symmetrizes the affinity matrix
        # np.save("data/affinity_matrix.npy", Affinity)
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


def get_random_orthogonal_matrix(D, d):
    # Returns a D x d orthogonal cost_matrix
    # Method inspired from https://stackoverflow.com/questions/38426349/how-to-create-random-orthonormal-matrix-in-python-numpy
    if D<d:
        print("D should be higher than d")
        import pdb; pdb.set_trace()
    random_matrix = H = np.random.randn(D, d)
    Q, r = np.linalg.qr(random_matrix, mode="reduced") # Mode = "reduced" to ensure that Q is of dimensions D x d
    return Q

def Lasso_minimization(data, mu2, tau):
    i = 0
    D, N = data.shape
    C = np.zeros((N, N))
    Gamma2 = np.zeros((N, N))
    XT_X = np.dot(data.T, data)
    converged = False
    while not converged:
        Z = np.dot(np.linalg.inv(tau*XT_X + mu2*np.identity(N)), tau*XT_X + mu2*(C - Gamma2/mu2))
        C = shrinkage(Z + Gamma2/mu2, 1/mu2)
        C = C - np.diag(np.diag(C))
        Gamma2 = Gamma2 + mu2*(Z-C)
        i += 1
        converged = (i > 50)
    return C

def shrinkage(X, tau):
    U, Sigma, VT = SVD(X)
    newSingularValue = []
    for i in np.diag(Sigma):
        if i > tau:
            i -= tau
        elif i < -tau:
            i += tau
        else:
            i = 0
        newSingularValue.append(i)
    return np.dot(U, np.dot(np.diag(newSingularValue), VT))

if __name__=="__main__":
    pict, labels = load_Yale_data()
    Lasso_minimization(pict[:100], 0.1, 5)
    pass
