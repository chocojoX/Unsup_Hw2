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

def load_Hopkins_data(dataset):
    if dataset == "cars2B":
        data = spio.loadmat("data/Hopkins155/cars2B/cars2B_truth.mat")
    elif dataset == "1R2RC":
        data = spio.loadmat("data/Hopkins155/1R2RC/1R2RC_truth.mat")
    elif dataset == "kanatani1":
        data = spio.loadmat("data/Hopkins155/kanatani1/kanatani1_truth.mat")
    else :
        raise Exception("Unknown dataset")
    n = max(data['s'])
    N = data['x'].shape[1]
    F = data['x'].shape[2]
    D = 2*F;
    pictures = np.reshape(data['x'][1:,:],(N,D))
    labels = data['s'] - 1
    labels = np.reshape(labels,(1, labels.shape[0]))
    return pictures.T, labels[0], n[0]


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


def compute_affinity_matrix(data, K, sigma, n_pictures, load_from_file=False, verbose=0, dataset_name= "Yale"):
    # data is D by N
    # K : number of closest neighbours
    # sigma parameter of the gaussian
    D, N = data.shape
    data_normalized = data/(np.linalg.norm(data, axis=0).reshape(1,-1))

    Affinity = np.zeros((n_pictures, n_pictures))
    distance_matrix = np.zeros((N, N))

    # Computes distance matrix

    matrix_path = str("data/distance_matrix_"+dataset_name+".npy")
    if os.path.exists(matrix_path):
        distance_matrix = np.load(matrix_path)
    else:
        print("Computing distance matrix, this may take a while but will only be performed once")

        t0 = time.time()
        data_total, _ = load_Yale_data()
        data_total = data_total/(np.linalg.norm(data_total, axis=0).reshape(1,-1))
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sum((data_total[:, i] - data_total[:, j])**2)
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        distance_matrix = cdist(data_total.T,data_total.T, 'sqeuclidean')
        t1 = time.time()
        print("Time to compute distance matrix : %.1f s" %(t1-t0))
        np.save(matrix_path, distance_matrix)

    # Computes affinity matrix
    matrix_path = str("data/affinity_matrix_"+dataset_name+".npy")
    if os.path.exists(matrix_path) and load_from_file:
        Affinity = np.load(matrix_path)
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
    data = data/(np.linalg.norm(data, axis=0).reshape(1,-1))
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
    X[np.abs(X) <= tau] = 0
    X[X >= tau] -= tau
    X[X <= -tau] += tau
    return X


if __name__=="__main__":
    pict, labels, n = load_Hopkins_data()
    print(pict.shape)
    print(labels)
    pass
