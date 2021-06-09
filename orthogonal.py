# PHS3350
# Week 14 - Orthogonality of eigenfunctions of a family of non-Hermitian Hamiltonians
# "what I cannot create I do not understand" - R. Feynman.
# Ana Fabela Hinojosa, 09/06/2021

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.special import gamma
import scipy.special as sc
from scipy import linalg
from tqdm import tqdm
from odhobs import psi as cpsi_blank

plt.rcParams['figure.dpi'] = 200
np.set_printoptions(linewidth=200)

def linear_algebra(matrix):
    eigenvalues_list = []
    eigenvectors_list = []
    for i in range(len(matrix)):
        eigenvalues, eigenvectors = linalg.eig(matrix)
        eigenvalues_list.append(eigenvalues)
        eigenvectors_list.append(eigenvectors)
    return np.array(eigenvalues_list), np.array(eigenvectors_list)

def filtering_and_sorting(evals, evects):
    mask = (0 < evals.real) & (evals.real < 20)
    # print(mask)
    evals = evals[mask]
    evects = evects[:, mask]

    # sorting
    order = np.argsort(np.round(evals.real, 3) + np.round(evals.imag, 3) / 1e6)
    # print(order)
    evals = evals[order]
    evects = evects[:, order]
    return evals, evects

def inner_product(evects):
    # print(len(evects))
    a = evects[:, 39]
    b = evects[:, 40]
    return np.vdot(a, b)

################################# GLOBALS ######################################

matrix_04 = np.load("matrix_060.npy")
# matrix_07 = np.load("matrix_030.npy")

############################## function calls ##################################

eigenvalues_04, eigenvectors_04 = linear_algebra(matrix_04)
eigenvalues_04, eigenvectors_04 = filtering_and_sorting(eigenvalues_04, eigenvectors_04)
dot_product = inner_product(eigenvectors_04)
print(dot_product)

# eigenvalues_07, eigenvectors_07 = linear_algebra(matrix_07)
# eigenvalues_list_07, eigenvectors_list_07 =  make_lists(eigenvalues_07, eigenvectors_07)
