import numpy as np
from munkres import Munkres, print_matrix


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
    pass
