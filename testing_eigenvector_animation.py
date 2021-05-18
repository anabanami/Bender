# PHS3350
# Week 5 - Energy levels of a family of non-Hermitian Hamiltonians
# "what I cannot create I do not understand" - R. Feynman.
# Ana Fabela Hinojosa, 16/05/2021

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.special import gamma
import scipy.special as sc
from scipy import linalg
from odhobs import psi as cpsi_blank

plt.rcParams['figure.dpi'] = 200
np.set_printoptions(linewidth=200)

####################### FINAL plot of eigenvalues ############################

Energies_2 = np.load("Energies_unbroken.npy")

def linear_algebra(epsilons):

    eigenvalues_list = []
    eigenvectors_list = []
    for i, ϵ in enumerate(epsilons):
        matrix = np.load(f'matrices/matrix_{i:03d}.npy')
        eigenvalues, eigenvectors = linalg.eig(matrix)
        eigenvalues_list.append(eigenvalues)
        eigenvectors_list.append(eigenvectors)

        ϵ_list = np.full(len(eigenvalues), ϵ)

    return np.array(eigenvalues_list), np.array(eigenvectors_list)

####################### sorting Eigenvectors ##################################

def filtering_and_sorting(evals, evects):
    mask = (0 < evals.real) & (evals.real < 20) & (evals.imag < 0.3)
    # print(mask)
    evals = evals[mask]
    evects = evects[:, mask]

    # sorting
    order = np.argsort(evals)
    # print(order)
    evals = evals[order]
    evects = evects[:, order]
    return evals[1:3], evects[:, 1:3]

####################### Eigenvectors plot ##################################

def spatial_wavefunctions(N, x, epsilons, evals, evects):
    # calculating basis functions
    x[x == 0] = 1e-200
    PSI_ns = []
    for n in range(N):
        psi_n = cpsi_blank(n, x)
        PSI_ns.append(psi_n)
    PSI_ns = np.array(PSI_ns)
    np.save(f"PSI_ns.npy", PSI_ns)


    for i, ϵ in enumerate(epsilons):

        _, c = filtering_and_sorting(evals[i], evects[i])

        if c.shape[1] < 2:
            print(i, "continuing")
            continue
        eigenstates = []
        for j in range(2):
            # print(len(evects))
            d = c[:, j]
            psi_jx = np.zeros(x.shape, complex)
            # for each H.O. basis vector relevant to the filtered and sorted eigenvectors
            
            for n in range(N):
                psi_jx += d[n] * PSI_ns[n]
                # normalise
                psi_jx /= np.sqrt(np.sum(abs(psi_jx) ** 2 * delta_x))
                # impose phase convention at ~ x = 0
                psi_jx *= np.exp(-1j * np.angle(psi_jx[Nx // 2]))

            eigenstates.append(psi_jx)

        plt.plot(x, np.real(eigenstates[0]), "-", color='blue', linewidth=0.5, label=r"Re($\psi_1$)")
        plt.plot(x, np.imag(eigenstates[0]), "--", color='blue', linewidth=0.5, label=r"Im($\psi_1$)")
        plt.plot(x, np.real(eigenstates[1]), "-", color='orange', linewidth=0.5, label=r"Re($\psi_2$)")
        plt.plot(x, np.imag(eigenstates[1]), "--", color='orange', linewidth=0.5, label=r"Im($\psi_2$)")
        plt.legend()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$ \psi_{n}$')
        plt.savefig(f"spatial_wavefunctions/wavefunction_{i:03d}.png")
        plt.clf()

            
####################### Function calls ##################################

N = 100
Nϵ = 100
Nx = 1024
epsilons = np.linspace(-1.0, 0, Nϵ)
xs = np.linspace(-10, 10, Nx)
delta_x = xs[1] - xs[0]

evals, evects = linear_algebra(epsilons)

spatial_wavefunctions(N, xs, epsilons, evals, evects)






