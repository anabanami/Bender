# PHS3350
# Week 12 - Testing HO states vs my calculated states
# "what I cannot create I do not understand" - R. Feynman.
# Ana Fabela Hinojosa, 25/05/2021

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import quad
import scipy.special as sc
from scipy import linalg
from tqdm import tqdm
from odhobs import psi as cpsi_blank


plt.rcParams['figure.dpi'] = 200
np.set_printoptions(linewidth=200)

############################## complex integration ###################################

# def complex_quad(func, a, b, **kwargs):
#     # Integration using scipy.integratequad() for a complex function
#     def real_func(*args):
#         return np.real(func(*args))

#     def imag_func(*args):
#         return np.imag(func(*args))

#     real_integral = quad(real_func, a, b, **kwargs)
#     imag_integral = quad(imag_func, a, b, **kwargs)
#     return real_integral[0] + 1j * imag_integral[0]

#################################### Matrix ###########################################

# def Hamiltonian(x, ϵ, n):
#     x = np.array(x)
#     x[x == 0] = 1e-200
#     h = 1e-6
#     psi_n = cpsi_blank(n, x)
#     d2Ψdx2 = (cpsi_blank(n, x + h) - 2 * psi_n + cpsi_blank(n, x - h)) / h ** 2
#     return -d2Ψdx2 + (x ** 2 * (1j * x) ** ϵ) * psi_n


# def element_integrand(x, ϵ, m, n):
#     psi_m = cpsi_blank(m, x)
#     return np.conj(psi_m) * Hamiltonian(x, ϵ, n)


# # NxN MATRIX
# def Matrix(x, N):
#     M = np.zeros((N, N), dtype="complex")
#     for m in tqdm(range(N)):
#         for n in tqdm(range(N)):
#             b = np.abs(np.sqrt(4 * min(m, n) + 2)) + 2
#             element = complex_quad(
#                 element_integrand, -b, b, args=(ϵ, m, n), epsabs=1.49e-08, limit=1000
#             )
#             # print(element)
#             M[m][n] = element
#     return M

#################################### Matrix ###########################################

def linear_algebra():

    matrix = np.load(f'matrix_ϵ0.npy')
    eigenvalues, eigenvectors = linalg.eig(matrix)

    return np.array(eigenvalues), np.array(eigenvectors.T)


def filtering_and_sorting(evals, evecs):

    # # filtering
    mask = (0 < evals.real) & (evals.real < 20)
    evals = evals[mask]
    evecs = evecs[:, mask]

    # # sorting
    order = np.argsort(np.round(evals.real, 3) + np.round(evals.imag, 3) / 1e6)
    # print(order)
    evals = evals[order]
    evecs = evecs[:, order]

    # print(evals)

    return evals[1:3], evecs[:, 1:3]


def spatial_wavefunctions(N, x):

    PSI_ns = np.load("PSI_ns.npy")

    c = eigenvectors
    eigenstates = []
    for j in range(2):
        d = c[:, j]
        psi_jx = np.zeros(x.shape, complex)
        # for each H.O. basis vector relevant to the filtered and sorted eigenvectors
        for n in range(N):
            psi_jx += d[n] * PSI_ns[n]
        # # impose phase convention at ~ x = 0
        psi_jx *= np.exp(-1j * np.angle(psi_jx[Nx // 2]))
        # # # normalise
        psi_jx /= np.sqrt(np.sum(abs(psi_jx) ** 2 * delta_x))

        eigenstates.append(psi_jx)


    fig, ax = plt.subplots()
    plt.plot(x, abs(eigenstates[0])**2, "-", color='blue', linewidth=1, label=fr"$|\psi_1|^2$") # first excited state
    plt.plot(x, abs(PSI_ns[1]) ** 2, "--", color='green', linewidth=0.5, label=r"$|\psi_{1, HO}|^2$") # first excited state HO
    # plt.plot(x, abs(eigenstates[1])**2, "-", color='orange', linewidth=1, label=fr"$|\psi_2|^2$") # second excited state
    plt.legend(loc="upper right")
    plt.xlabel(r'$x$')
    plt.ylabel(r'$ \psi_{n}$')
    textstr = '\n'.join((
        fr'$E_1 = {eigenvalues[0]:.03f}$',
        fr'$E_2 = {eigenvalues[1]:.03f}$',
        fr'$ϵ = {ϵ:.03f}$'
        ))

    # place a text box in upper left in axes coords
    ax.text(0.02, 1.15, textstr, transform=ax.transAxes, verticalalignment='top')
    plt.show()
    plt.savefig(f"spatial_wavefunction_ϵ0.png")
    plt.clf()


###################################function calls################################################
# GLOBALS
x = 2
ϵ = 0
N = 100
Nx = 1024
xs = np.linspace(-20, 20, Nx)
delta_x = xs[1] - xs[0]

# NxN MATRIX for ϵ = 0
# matrix_ϵ0 = Matrix(x, N)
# np.save('matrix_ϵ0.npy', matrix_ϵ0)

eigenvalues, eigenvectors = linear_algebra()

eigenvalues, eigenvectors = filtering_and_sorting(eigenvalues, eigenvectors)

spatial_wavefunctions(N, xs)
