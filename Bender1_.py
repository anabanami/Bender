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

plt.rcParams['figure.dpi'] = 200

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
    # call: complex_fsolve(error, E0, args=(ϵ, n))
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

def integrand(x_prime, tp_minus, E, ϵ): 
    # Change of variables integrand
    x = x_prime + 1j * np.imag(tp_minus)
    return np.sqrt(E - x**2 * (1j * x)**ϵ)

def LHS(n):
    # Quantization condition
    return (n + 1/2) * np.pi

def RHS(E, ϵ): #***********************************************************************************
    # Integral defining E
    # integration LIMITS
    tp_minus = E**(1/(ϵ+2)) * np.exp(1j * np.pi * (3/2 - (1/(ϵ+2))))
    tp_plus = E**(1/(ϵ+2)) * np.exp(-1j * np.pi * (1/2 - (1/(ϵ+2))))
    tp_minus_prime = np.real(E**(1/(ϵ+2)) * np.exp(1j * np.pi * (3/2 - (1/(ϵ+2)))) - 1j * np.imag(tp_minus))
    tp_plus_prime = np.real(E**(1/(ϵ+2)) * np.exp(-1j * np.pi * (1/2 - (1/(ϵ+2)))) - 1j * np.imag(tp_minus)) 
    # print(tp_minus_prime)
    # print(tp_plus_prime)
    return complex_quad(integrand, tp_minus_prime, tp_plus_prime, args=(tp_minus, E, ϵ))

def error(E, ϵ, n):
    return RHS(E, ϵ) - LHS(n)

def compare():
    # comparison to WKB results reported for  (ϵ, n) = (1, 0) using IC: E0 = E0
    E_RK = E0
    E_WKB = 1.0943
    diff_RK_WKB = E_RK - E_WKB
    diff_RK_mine = E_RK - complex_fsolve(error, E0, args=(1, 0))
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
    tp_minus = E**(1/(ϵ+2)) * np.exp(1j * np.pi * (3/2 - (1/(ϵ+2))))
    tp_plus = E**(1/(ϵ+2)) * np.exp(-1j * np.pi * (1/2 - (1/(ϵ+2))))
    tp_minus_prime = E**(1/(ϵ+2)) * np.exp(1j * np.pi * (3/2 - (1/(ϵ+2)))) - 1j * np.imag(tp_minus) #is this + or -?
    tp_plus_prime = E**(1/(ϵ+2)) * np.exp(-1j * np.pi * (1/2 - (1/(ϵ+2)))) - 1j * np.imag(tp_minus) 
    # domain & differential (infinitesimal)
    x_prime = np.linspace(tp_minus_prime, tp_plus_prime, 50000)
    dx_prime = x_prime[1] - x_prime[0]
    return np.sum(real_func(x_prime, E, ϵ) * dx_prime)


#######################################function calls########################################################

#IC based on RK results give (ϵ, n) = (1, 0)
# ϵ = 1
E0 = 1.1563
# E = E0

ϵ = 2
E = 60.185767651 

tp_minus = E**(1/(ϵ+2)) * np.exp(1j * np.pi * (3/2 - (1/(ϵ+2))))
tp_plus = E**(1/(ϵ+2)) * np.exp(-1j * np.pi * (1/2 - (1/(ϵ+2))))
tp_minus_prime = E**(1/(ϵ+2)) * np.exp(1j * np.pi * (3/2 - (1/(ϵ+2)))) - 1j * np.imag(tp_minus)
tp_plus_prime = E**(1/(ϵ+2)) * np.exp(-1j * np.pi * (1/2 - (1/(ϵ+2)))) - 1j * np.imag(tp_minus)

########################## TEST 2 #######################################
# ITERATIVE approach 1 for ϵ = 1
Energies_1 = []
for n in range(10):
  E = complex_fsolve(error, E0, args=(1, n))
  Energies_1.append(E)

# comparison to WKB and RK reported in Bender 
n = range(10)
E_RK= [1.1563, 4.1093, 7.5623, 11.3144, 15.2916, 19.4515, 23.7667, 28.2175, 32.7891, 37.4698]
E_WKB = [1.0943, 4.0895, 7.5489, 11.3043, 15.2832, 19.4444, 23.7603, 28.2120, 32.7841, 37.4653]
plt.plot(n, Energies_1 , label="my calculated energies")
plt.plot(n, E_RK , label=r"$E_{RK}$")
plt.plot(n, E_WKB , label=r"$E_{WKB}$")
plt.legend()
plt.xlabel("n")
plt.ylabel("E")
plt.show()

# ITERATIVE approach 2
Energies_2 = []
for n in range(10):
    E_s = []
    for ϵ in np.linspace(0, 3, 30):
        E = complex_fsolve(error, E0, args=(ϵ, n))
        E_s.append(E)
        # print(f" {ϵ = }, {n = }, {E = }")
    Energies_2.append(E_s)
# print(Energies_2)

#PLOTING ITERATIVE approach 1
for E_ϵs in Energies_2:
    # print(E_ϵs)
    ϵ = np.linspace(0, 3, 30)
    plt.plot(ϵ, E_ϵs, "o-", markersize=2)
# plt.legend()
plt.ylim(0, 20)
plt.xlabel("ϵ")
plt.ylabel("E")
plt.show()

########################## TEST 2 #######################################

##########################TEST#######################################
# # FIND ENERGIES TO FEED INTO BRUTE INTEGRAL
# Energies = []
# # i = 0
# for ϵ in range(3):
#     # print(f"{ϵ = }")
#     E_s = []
#     for n in range(10):
#         E = complex_fsolve(error, E0, args=(ϵ, n))
#         E_s.append(E)
#         # print(f"{i = }, {n = }, {E = }")
#         # i+=1
#     Energies.append(E_s)

# qc = (6 + 1/2) * np.pi
# brute_integral = brute_force(integrand, Energies[1][6], 1)

# print("\nCase: (ϵ, n) = (1,6)")
# print(f"\n(n + 1/2)π = {qc:.04f}")
# print(f"\nEnergy from complex_fsolve() E = {Energies[1][6]:.04f}")
# print(f"\nEnergy from {brute_integral = :.04f}\n")

# # ITERATIVE approach 1
# Energies = []
# for n in range(10):
#     E_s = []
#     for ϵ in np.linspace(0, 3, 30):
#         E = complex_fsolve(error, E0, args=(ϵ, n))
#         E_s.append(E)
#         # print(f" {ϵ = }, {n = }, {E = }")
#     Energies.append(E_s)
# # print(f"{Energies = }")

# #PLOTING ITERATIVE
# for E_ϵs in Energies:
#     # print(E_ϵs)
#     ϵ = np.linspace(0, 3, 30)
#     plt.plot(ϵ, E_ϵs, "o-", markersize=3)#, label="mine")
# # plt.legend()
# plt.ylim(0, 20)
# plt.xlabel("ϵ")
# plt.ylabel("E")
# plt.show()
##########################TEST#######################################

##########################TEST#######################################
## Single case Comparison to Bender's analytic result (ϵ, n)
# CHANGE n
# n = 2
# E_ϵs_n = []
# analytic_E_ϵs_n = []
# for ϵ in np.linspace(0, 3, 30):
#     # complex_fsolve(error, E0, args=(1, 0))?
#     E_ϵ_n = complex_fsolve(error, E0, args=(ϵ, n))
#     E_ϵs_n.append(E_ϵ_n)

#     analytic_E_ϵ_n = analytic_E(ϵ, n)
#     analytic_E_ϵs_n.append(analytic_E_ϵ_n)

# ϵ = np.linspace(0, 3, 30)
# plt.plot(ϵ, E_ϵs_n, "o-", markersize=2, label="Ana")
# plt.plot(ϵ, analytic_E_ϵs_n, "o-", markersize=2, label="Bender's")
# plt.legend()
# plt.ylim(0, 20)
# plt.xlabel("ϵ")
# plt.ylabel("E")
# plt.show()

##########################TEST#######################################

# ######################### TEST 1 #######################################
# def whats_up_with_integrand(x_values, E, ϵ):
#     # checking the integration path of the integrand in the x-complex plane
#     reals = []
#     imaginary = []
#     for x in x_values:
#         complex_num = np.sqrt(E - x**2 * (1j * x)**ϵ)

#         reals.append(np.real(complex_num))
#         imaginary.append(np.imag(complex_num))

#     plt.plot(reals, imaginary, '-')
#     plt.plot(reals[0], imaginary[0],'go', label='start here')
#     plt.plot(reals[-1], imaginary[-1],'ro', markersize='1.2', label='finish here')
#     plt.legend()
#     plt.ylabel(r'$Im(\sqrt{E - x^2 (i x)^\epsilon})$')
#     plt.xlabel(r'$Re(\sqrt{E - x^2 (i x)^\epsilon})$')

#     # plt.title("Bender's integration contour")
#     plt.title("change or variables integration contour")
#     plt.show()


# # Bender's integral
# x_values = np.linspace(tp_minus, tp_plus, 10000)

# # change of variables integral 
# x_values = np.linspace(tp_minus_prime, tp_plus_prime, 10000) + 1j * np.imag(tp_minus)

# whats_up_with_integrand(x_values, E0, ϵ)
# ######################### TEST 1 #######################################