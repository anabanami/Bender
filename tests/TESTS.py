##########################TEST########################################
## iterative approach for ϵ = 0 case:
# Energies = []
# for n in range(10):
#   E = complex_fsolve(error, 1.1563, args=(0, n))
#   Energies.append(E)
  # print(f"n = {n}, E = {E:.04f}")
##########################TEST#######################################

##########################TEST#######################################
# Is = []
# for E in np.linspace(0, 15, 100):
#     I = RHS(E, 1)
#     Is.append(I)

# E = np.linspace(0, 15, 100)
# plt.plot(E, np.real(Is), label="real part")
# plt.axhline(np.pi *0.5, linestyle='--', label="π/2")
# plt.axvline(1.09, linestyle=':', color='r', label=r"expected $E_{WKB}$")
# plt.legend()
# plt.xlabel("E")
# plt.ylabel("Re(I(E, ϵ))")
# plt.show()
##########################TEST#######################################

##########################TEST#######################################
## *******FIND ENERGIES TO FEED INTO BRUTE INTEGRAL ***************
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
##########################TEST#######################################

##########################TEST#######################################
## ***Single case Comparison to Bender's analytic result (ϵ, n) ***
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

########################## TEST 1 #######################################
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
# # x_values = np.linspace(tp_minus, tp_plus, 10000)

# # change of variables integral 
# x_values = np.linspace(tp_minus_prime, tp_plus_prime, 10000) + 1j * np.imag(tp_minus)

# whats_up_with_integrand(x_values, E, ϵ)
########################## TEST 1 #######################################


########################## TEST 2 #######################################
# # ITERATIVE approach 1 for ϵ = 1
# Energies_1 = []
# for n in range(10):
#   E = complex_fsolve(error, E0, args=(1, n))
#   Energies_1.append(E)

# # comparison to WKB and RK reported in Bender 
# n = range(10)
# E_RK= [1.1563, 4.1093, 7.5623, 11.3144, 15.2916, 19.4515, 23.7667, 28.2175, 32.7891, 37.4698]
# E_WKB = [1.0943, 4.0895, 7.5489, 11.3043, 15.2832, 19.4444, 23.7603, 28.2120, 32.7841, 37.4653]
# plt.plot(n, Energies_1 , label="my calculated energies")
# plt.plot(n, E_RK , label=r"$E_{RK}$")
# plt.plot(n, E_WKB , label=r"$E_{WKB}$")
# plt.legend()
# plt.xlabel("n")
# plt.ylabel("E")
# plt.show()

# # ITERATIVE approach 2
# Energies_2 = []
# for n in range(10):
#     E_s = []
#     for ϵ in np.linspace(0, 3, 30):
#         E = complex_fsolve(error, E0, args=(ϵ, n))
#         E_s.append(E)
#         # print(f" {ϵ = }, {n = }, {E = }")
#     Energies_2.append(E_s)
# # print(Energies_2)

# #PLOTING ITERATIVE approach 1
# for E_ϵs in Energies_2:
#     # print(E_ϵs)
#     ϵ = np.linspace(0, 3, 30)
#     plt.plot(ϵ, E_ϵs, "o-", markersize=2)
# # plt.legend()
# plt.ylim(0, 20)
# plt.xlabel("ϵ")
# plt.ylabel("E")
# plt.show()

# ########################## TEST 2 #######################################





## If fsolve was less smart... I should use this
# for n in range(15):
#     for ϵ in np.linspace(-1, 3, 512):
#         def err(E):
#             return error(E, ϵ, n)

