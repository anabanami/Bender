# PHS3350
# Week 5 - Energy levels of a family of non-Hermitian Hamiltonians
# "what I cannot create I cannot understand" - R. Feynman. 
# Ana Fabela Hinojosa, 04/04/2021

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.special import gamma

plt.rcParams['figure.dpi'] = 150

def complex_quad(func, a, b, **kwargs):
    # Integration using scipy.integratequad() for a complex function
    def real_func(*args):
        return np.real(func(*args))
    def imag_func(*args):
        return np.imag(func(*args))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return real_integral[0] + 1j*imag_integral[0]

## TEST
def complex_fsolve(func, E0, **kwargs):
    # root finding algorithm. FINDS: Energy values from error() function
    # call: complex_fsolve(error, 1.1563, args=(ϵ, n))
    def real_func(*args):
        return np.real(func(*args))
    real_root = fsolve(real_func, E0, **kwargs)
#     def imag_func(*args):
#         return np.imag(func(*args))
#     imag_root = fsolve(imag_func, E0, **kwargs)
    # Check that the imaginary part of func is also zero
    value = func(real_root[0], *kwargs['args'])
    assert abs(np.imag(value)) < 1e-10, "Imaginary part wasn't zero"
    # print(f"E = {Energies[0]:.04f}")
    return real_root[0]

def integrand(x_prime, E, ϵ):
    # Change of variables
    α = 3/2 - 1/(ϵ +2)
    offset = - 1j * np.sin(np.pi * (3/2 - 1/(ϵ +2)))
    x = x_prime - offset
    return np.sqrt(E - x**2 * (1j * x)**ϵ)

def RHS(E, ϵ):
    # Integral defining E
    offset = - 1j * np.sin(np.pi * (3/2 - 1/(ϵ +2)))
    # Change of variables
    tp_minus_prime = E**(1/(ϵ+2)) * np.cos(np.pi * (3/2 - (1/(ϵ+2))))
    tp_plus_prime = E**(1/(ϵ+2)) * np.cos(np.pi * (1/2 - (1/(ϵ+2))))
    return complex_quad(integrand, tp_minus_prime, tp_plus_prime, args=(E, ϵ))

def LHS(n):
    # Quantization condition
    return (n +1/2) * np.pi

def error(E, ϵ, n):
    return RHS(E, ϵ) - LHS(n)

def compare():
    # comparison to WKB results reported for  (ϵ, n) = (1, 0) using E0 = 1.1563
    E_RK = 1.1563
    E_WKB = 1.0943
    diff_RK_WKB = E_RK - E_WKB
    diff_RK_mine = E_RK - complex_fsolve(error, 1.1563, args=(1, 0))
    how_many_sigmas_theory = diff_RK_WKB / E_RK
    how_many_sigmas_mine = diff_RK_mine / E_RK
    print(f"\nnumber of 𝞼 away is WKB from exact result: {abs(how_many_sigmas_theory):.3f}")
    print(f"number of 𝞼 away am I from exact result: {abs(how_many_sigmas_mine):.3f}\n")

def analytic_E(ϵ, n):
    # Bender equation (34) pg. 960
    top = gamma(3/2 + 1/(ϵ +2)) * np.sqrt(np.pi) * (n + 1/2)
    bottom = np.sin(np.pi / (ϵ + 2)) * gamma(1 + 1/(ϵ + 2))
    return (top/bottom)**((2 * ϵ + 4)/(ϵ + 4))

def brute_force(func, E, ϵ):
    # BRUTE INTEGRAL
    def real_func(*args):
        return np.real(func(*args))
    # limits
    tp_minus_prime = E**(1/(ϵ+2)) * np.cos(np.pi * (3/2 - (1/(ϵ+2))))
    tp_plus_prime = E**(1/(ϵ+2)) * np.cos(np.pi * (1/2 - (1/(ϵ+2))))
    # domain & differential (infinitesimal)
    x_prime = np.linspace(tp_minus_prime, tp_plus_prime, 50000)
    dx_prime = x_prime[1] - x_prime[0]
    return np.sum(real_func(x_prime, E, ϵ) * dx_prime)

# FIND ENERGIES TO FEED INTO BRUTE INTEGRAL
Energies = []
# i = 0
for ϵ in range(3):
    # print(f"{ϵ = }")
    E_s = []
    for n in range(10):
        E = complex_fsolve(error, 1.1563, args=(ϵ, n))
        E_s.append(E)
        # print(f"{i = }, {n = }, {E = }")
        # i+=1
    Energies.append(E_s)

qc = (6 + 1/2) * np.pi
brute_integral = brute_force(integrand, Energies[1][6], 1)

print("\nCase: (ϵ, n) = (1,6)")
print(f"\n(n + 1/2)π = {qc:.04f}")
print(f"\nEnergy from complex_fsolve() E = {Energies[1][6]:.04f}")
print(f"\nEnergy from {brute_integral = :.04f}\n")

# ITERATIVE approach 1
# Energies = []
# for n in range(10):
#     E_s = []
#     for ϵ in np.linspace(0, 3, 30):
#         E = complex_fsolve(error, 1.1563, args=(ϵ, n))
#         E_s.append(E)
#         print(f" {ϵ = }, {n = }, {E = }")
#     Energies.append(E_s)
# # print(f"{Energies = }")

# #PLOTING ITERATIVE
# for E_ϵs in Energies:
#     # print(E_ϵs)
#     ϵ = np.linspace(0, 3, 30)
#     plt.plot(ϵ, E_ϵs, "o-", markersize=3, label="mine")
# plt.legend()
# plt.ylim(0, 20)
# plt.xlabel("ϵ")
# plt.ylabel("E")
# plt.show()

##########################TEST#######################################
## Single case Comparison to Bender's analytic result (ϵ, n)
# CHANGE n
# n = 2
# E_ϵs_n = []
# analytic_E_ϵs_n = []
# for ϵ in np.linspace(0, 3, 30):
#     # complex_fsolve(error, 1.1563, args=(1, 0))?
#     E_ϵ_n = complex_fsolve(error, 1.1563, args=(ϵ, n))
#     E_ϵs_n.append(E_ϵ_n)

#     analytic_E_ϵ_n = analytic_E(ϵ, n)
#     analytic_E_ϵs_n.append(analytic_E_ϵ_n)

# ϵ = np.linspace(0, 3, 30)
# plt.plot(ϵ, E_ϵs_n, "o-", markersize=2, label="mine")
# plt.plot(ϵ, analytic_E_ϵs_n, "o-", markersize=2, label="Bender's")
# plt.legend()
# plt.ylim(0, 20)
# plt.xlabel("ϵ")
# plt.ylabel("E")
# plt.show()
##########################TEST#######################################