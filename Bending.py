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

def complex_quad(func, a, b, **kwargs):
    # Integration using scipy.integratequad() for a complex function
    def real_func(*args):
        return np.real(func(*args))

    def imag_func(*args):
        return np.imag(func(*args))

    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return real_integral[0] + 1j * imag_integral[0]

## Runge-Kutta, finding IC!
def find_k(x, ϵ, E):
    return np.sqrt(x ** 2 * (1j * x) ** ϵ - E)

def abs_clip(x, level):
    if abs(x) > level:
        return level * x / abs(x)
    else:
        return x

# Schrödinger equation
def Schrodinger_eqn(x, Ψ):
    psi, psi_prime = Ψ
    psi_primeprime = (x ** 2 * (1j * x) ** ϵ - E) * psi
    Ψ_prime = np.array([psi_prime, psi_primeprime])
    return Ψ_prime

def Runge_Kutta(x, delta_x, E, ϵ, Ψ):
    k1 = Schrodinger_eqn(x, E, ϵ, Ψ)
    k2 = Schrodinger_eqn(x + delta_x / 2, E, ϵ, Ψ + k1 * delta_x / 2)
    k3 = Schrodinger_eqn(x + delta_x / 2, E, ϵ, Ψ + k2 * delta_x / 2)
    k4 = Schrodinger_eqn(x + delta_x, E, ϵ, Ψ + k3 * delta_x)
    return Ψ + (delta_x / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

#################################### Matrix SOLVING ###########################################         )

def Hamiltonian(x, ϵ, n):
    x = 1e-200
    # x[x == 0] = 1e-200
    h = 1e-6
    psi_n = cpsi_blank(n, x)
    d2Ψdx2 = (cpsi_blank(n, x + h) - 2 * psi_ni + cpsi_blank(n, x - h)) / h ** 2
    return d2Ψdx2 + (x ** 2 * (1j * x) ** ϵ) * psi_n

def element_integrand(x, ϵ, m, n):
    # CHECK THESE IF mass = 1 instead of 1/2
    psi_m = cpsi_blank(m, x)
    return np.conj(psi_m) * Hamiltonian(x, ϵ, n)

# NxN MATRIX
def Matrix(x, N):
    M = np.zeros((N, N), dtype="complex")
    for m in tqdm(range(N)):
        for n in tqdm(range(N)):
            b = np.abs(np.sqrt(4 * min(m, n) + 2)) + 2
            element = complex_quad(
                element_integrand, -b, b, args=(ϵ, m, n), epsabs=1.49e-08, limit=1000
            )
            # print(element)
            M[m][n] = element
    return M

###################################function calls################################################
# GLOBALS
epsilons = np.linspace(-1, 0, 5)
k = 1 / 2
x = 2
N = 800

# NxN MATRIX
for ϵ in epsilons:
    print(f"{ϵ = }")
    matrix = Matrix(x, N)
    np.save(f'matrix_{ϵ}.npy', matrix)
    # print(f"\nMatrix\n{matrix}")

# eigenvalues, eigenvectors = linalg.eig(Matrix)
# print(f"\nEigenvalues\n{eigenvalues}")
# print(f"\nEigenvectors\n{eigenvectors.round(10)}\n")
## print(np.sum(abs(eigenvectors**2), axis=0)) # eigenvectors are unitary?

##############plots################
# m = 300
# n = 300
# b = np.abs(np.sqrt(4 * min(m, n) + 2)) + 2
# xs = np.linspace(-40, 40, 2048 * 10, endpoint=False)
# plt.plot(xs, cpsi_blank(300, xs), linewidth=1)
# plt.plot(xs, np.real(Hamiltonian(xs, ϵ, n)), label="Real part", linewidth=1)
# plt.plot(xs, np.imag(Hamiltonian(xs, ϵ, n)), label="Imaginary part", linewidth=1)
# plt.plot(xs, np.real(element_integrand(xs, ϵ, m, n)), label="Real part", linewidth=1)
# plt.plot(xs, np.imag(element_integrand(xs, ϵ, m, n)), label="Imaginary part", linewidth=1)
# plt.axvline(b, color='grey' , linestyle=":", label="Turning points")
# plt.axvline(-b, color='grey' , linestyle=":")
# plt.legend()
# plt.show()
# ##############plots################



