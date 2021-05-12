# PHS3350
# Week 10 - Energy levels of a family of non-Hermitian Hamiltonians
# "what I cannot create I do not understand" - R. Feynman. 
# Ana Fabela Hinojosa, 11/05/2021

import numpy as np
from scipy import linalg
import matplotlib
import matplotlib.pyplot as plt

epsilons = np.linspace(-1.0, 0, 100)

for ϵ in epsilons:
    matrix = np.load(f'matrix_{ϵ}.npy')
    eigenvalues, blergh_eigenvectors = linalg.eig(matrix)

    positive_evals = [i for i in eigenvalues if 0 < np.real(i) < 15 and abs(np.imag(i)) < 1]
    sorted_eigenvalues = sorted(positive_evals, key=lambda x: np.real(x))

    ϵ_list = np.full(len(sorted_eigenvalues), ϵ)
    print(f"{ϵ_list = }")

    plt.plot(ϵ_list, np.real(sorted_eigenvalues), marker='o', linestyle='None', color='k', markersize=6)
    plt.plot(ϵ_list, np.imag(sorted_eigenvalues), marker='o', linestyle='None', color='r', markersize=6)

plt.axis(ymin=0, ymax=15)
plt.show()




