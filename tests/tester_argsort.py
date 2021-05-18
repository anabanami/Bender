import numpy as np
np.set_printoptions(linewidth=200)


evals = np.array([16, 3, 22, 1, 25, -6, 4, 64, 81, 45, 0, 25, 98, 20, 67, 2])
evects = np.array([[ 5, 7, 12, 1, 4, 7, 89, 22, 12, 13, 90, 23, 44, 66, 11, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])

def filtering_sorting(evals, evects):
    mask = (0 < evals.real) & (evals.real < 20)
    # print(mask)
    evals = evals[mask]
    evects = evects[:, mask]

    # sorting
    order = np.argsort(evals)
    # print(order)
    evals = evals[order]
    evects = evects[:, order]
    return evals, evects

eigenvalues, eigenvectors = filtering_sorting2(evals, evects)
print(eigenvalues)
print(eigenvectors)