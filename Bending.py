import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import quad
import scipy.special as sc
from scipy import linalg
from tqdm import tqdm


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

#################################### Matrix SOLVING ###########################################

def psi_blank(x, n):

    pi_powered = np.pi ** (-0.25)

    x = np.array(x)
    if n == 0:
        return np.ones_like(x) * pi_powered * np.exp(-x ** 2 / 2)
    if n == 1:
        return (
            np.sqrt(2.0)
            * x
            * np.pi ** (-0.25)
            * np.exp(-x ** 2 / 2)
        )
    h_i_2 = np.ones_like(x) * pi_powered
    h_i_1 = np.sqrt(2.0) * x * pi_powered
    sum_log_scale = np.zeros_like(x)
    for i in range(2, n + 1):

        # RECURRENCE RELATION
        h_i = np.sqrt(2.0 / i) * x * h_i_1 - np.sqrt((i - 1.0) / i) * h_i_2
        h_i_2, h_i_1 = h_i_1, h_i

        x[x == 0] = 1e-200

        log_scale = np.log(abs(h_i)).round()
        scale = np.exp(-log_scale)
        h_i = h_i * scale
        h_i_1 = h_i_1 * scale
        h_i_2 = h_i_2 * scale
        sum_log_scale += log_scale
    return h_i * np.exp(-x ** 2 / 2 + sum_log_scale)

# def psi_blank(x, n):
#     if n < 100:
#         return (
#             (1 / (2 ** n * sc.factorial(n)))
#             * (1 / (2 * np.pi)) ** (1 / 4)
#             * np.exp(-x ** 2 / 4)
#             * sc.eval_hermite(n, np.sqrt(1 / 2) * x)
#         )
#     else:
#         return (
#             2 ** (-1 / 2 * (n + 3 / 2))
#             * np.pi ** -3
#             / 4
#             * n ** (-1 / 2 - n)
#             * np.exp(n - (x ** 2) / 4)
#             * sc.eval_hermite(n, np.sqrt(1 / 2) * x)
#         )

def Hamiltonian(x, ϵ, n):
    h = 1e-6
    d2Ψdx2 = (psi_blank(x + h, n) - 2 * psi_blank(x, n) + psi_blank(x - h, n)) / h ** 2
    return d2Ψdx2 + (x ** 2 * (1j * x) ** ϵ) * psi_blank(x, n)

def element_integrand(x, ϵ, m, n):
    # CHECK THESE IF mass = 1 instead of 1/2
    psi_m = psi_blank(x, m)
    return np.conj(psi_m) * Hamiltonian(x, ϵ, n)

# NxN MATRIX
def Matrix(x, N):
    M = np.zeros((N, N), dtype="complex")
    for m in tqdm(range(N)):
        for n in tqdm(range(N)):
            b = np.abs(np.sqrt(4 * min(m, n) + 2)) + 2
            element = complex_quad(
                element_integrand, -b, b, args=(ϵ, m, n), epsabs=1.49e-02, limit=1000
            )
            # print(element)
            M[m][n] = element
    return M

###################################function calls################################################
# GLOBALS
ϵ = 1
k = 1 / 2
x = 2
N = 130

# NxN MATRIX
Matrix = Matrix(x, N)
np.save('matrix.npy', Matrix)
print(f"\nMatrix\n{Matrix}")

# eigenvalues, eigenvectors = linalg.eig(Matrix)
# print(f"\nEigenvalues\n{eigenvalues}")
# print(f"\nEigenvectors\n{eigenvectors.round(10)}\n")
## print(np.sum(abs(eigenvectors**2), axis=0)) # eigenvectors are unitary?

##############plots################
# xs = np.linspace(-40, 40, 2048 * 10, endpoint=False)
# # plt.plot(xs, psi_blank(xs, 300), linewidth=1)
# # plt.plot(xs, np.real(Hamiltonian(xs, ϵ, 300)), label="Real part", linewidth=1)
# # plt.plot(xs, np.imag(Hamiltonian(xs, ϵ, 300)), label="Imaginary part", linewidth=1)

# m = 300
# n = 300
# b = np.abs(np.sqrt(4 * min(m, n) + 2)) + 2

# plt.plot(xs, np.real(element_integrand(xs, ϵ, m, n)), label="Real part", linewidth=1)
# plt.plot(xs, np.imag(element_integrand(xs, ϵ, m, n)), label="Imaginary part", linewidth=1)
# plt.axvline(b, color='grey' , linestyle=":", label="classical turning points")
# plt.axvline(-b, color='grey' , linestyle=":")
# plt.legend()
# plt.show()
##############plots################


