# PHS3350
# Week 10 - Energy levels of a family of non-Hermitian Hamiltonians
# "what I cannot create I do not understand" - R. Feynman. 
# Ana Fabela Hinojosa, 11/05/2021

import numpy as np
from scipy import linalg
import matplotlib
import matplotlib.pyplot as plt

N = 100
epsilons = np.linspace(-1.0, 0, N)

for i, ϵ in enumerate(epsilons):
    matrix = np.load(f'matrices/matrix_{i:03d}.npy')
    eigenvalues, blergh_eigenvectors = linalg.eig(matrix)

    positive_evals = [i for i in eigenvalues if 0 < np.real(i) < 20 and abs(np.imag(i)) < 1]
    sorted_eigenvalues = sorted(positive_evals, key=lambda x: np.real(x))

    ϵ_list = np.full(len(sorted_eigenvalues), ϵ)
    # print(f"{ϵ_list = }")

    plt.plot(ϵ_list, np.real(sorted_eigenvalues), marker='.', linestyle='None', color='k', markersize=3)
    plt.plot(ϵ_list, np.imag(sorted_eigenvalues), marker='.', linestyle='None', color='r', markersize=3)

plt.axis(ymin=-1, ymax=20)
plt.axis()
plt.show()



