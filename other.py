import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.misc import derivative
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

plt.rcParams['figure.dpi'] = 200

## Runge-Kutta, finding IC!
def find_k(x, ϵ, E):
    return np.sqrt(x**2 * (1j * x)**ϵ - E)

def abs_clip(x, level):
    if abs(x) > level:
        return level * x / abs(x)
    else:
        return x

# # Schrödinger equation
# def Schrodinger_eqn(x, E, ϵ, Ψ):
#     u, v =  Ψ
#     v = abs_clip(v, 1e100)
#     w = (x**2 * (1j * x)**ϵ - E) * u
#     Ψ_prime = np.array([v, w])
#     return Ψ_prime

# Schrödinger equation
def Schrodinger_eqn(x, E, ϵ, Ψ):
    u, v =  Ψ
    # if derivative is blowing up return scaled version of derivative
    if abs(v) > 1e100:
        # print("clipping blow up!")
        v = abs_clip(v, 1e100)
        w = 0
        Ψ_prime = np.array([v, w])
        return Ψ_prime
    else:
        w = (x**2 * (1j * x)**ϵ - E) * u
        Ψ_prime = np.array([v, w])
        return Ψ_prime

def Runge_Kutta(x, delta_x, E, ϵ, Ψ):
    k1 = Schrodinger_eqn(x, E, ϵ, Ψ)
    k2 = Schrodinger_eqn(x + delta_x / 2, E, ϵ, Ψ + k1 * delta_x / 2) 
    k3 = Schrodinger_eqn(x + delta_x / 2, E, ϵ, Ψ + k2 * delta_x / 2) 
    k4 = Schrodinger_eqn(x + delta_x, E, ϵ, Ψ + k3 * delta_x) 
    return Ψ + (delta_x / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

############################### GLOBALS ###########################################
n = 100
# IC values from Bender table 1.
ϵ = 1
E = 1.1563

# Asymtotic x value for finding k
x_asymptotic = -30
# x values
xs = np.linspace(x_asymptotic, - x_asymptotic, n)
delta_x = xs[1] - xs[0]

# k values
k = find_k(x_asymptotic, ϵ, E)
# IC
Ψ0 = [1, 1j * k]

############################### SHOOTING method ##################################
# INITIAL ENERGY BOUNDS
E_lower = 0
E_upper = 20
Energy = np.linspace(E_lower, E_upper, n)

mod_squared_values = []
mod_squared_solutions = []

# first run of shooting method
for E_i in Energy:
    mod_squared_psi = []
    Ψ = Ψ0
    for x in xs:
        mod_squared_psi.append(abs(Ψ[0]**2))
        Ψ = Runge_Kutta(x, delta_x,  E_i, ϵ, Ψ)

    mod_squared_values.append(mod_squared_psi[-1])
    mod_squared_solutions.append(mod_squared_psi)


# print(f"\n{len(mod_squared_solutions) = }\n")
# print(f"\n{mod_squared_solutions = }\n")
# print(f"\n{len(mod_squared_psi) = }\n")
# print(f"\n{mod_squared_psi = }\n")
print(f"\n{len(mod_squared_values) = }\n")
print(f"\n{mod_squared_values = }\n")

plt.plot(xs, mod_squared_solutions[0], label=r"$E_0$")
plt.plot(xs, mod_squared_solutions[1], label=r"$E_1$")
plt.plot(xs, mod_squared_solutions[2], label=r"$E_2$")
plt.ylabel(r"$|u^2|$")
plt.xlabel("x")
plt.legend()
plt.show()



# ## after first run...
# ## step 5 in log book
# E_flip_signs = []
# E_flip = []
# for i in range(len(signs) - 1):
#     if not signs[i] == signs[i+1]:
#         E_flip.append([Es[i], Es[i+1]])
#         E_flip_signs.append([signs[i], signs[i+1]])
#     else:
#         continue

# # print(f"\n{E_flip=}\n")
# # print(f"\n{E_flip_signs=}\n")

# for i in range(len(E_flip)):
#     E_lower, E_upper = E_flip[i]
#     sign_lower, sign_upper = E_flip_signs[i]

# print(f"\n{sign_lower = } {sign_upper =}\n")

    # print("Hello, this is the start of the while loop")
    # while abs(E_upper - E_lower) > 1e-12 * meV:
    #     E_mid = (E_lower + E_upper) / 2
    #     print(f"{E_mid = }")
        # w = np.real(Runge_Kutta(x, E_i, delta_x, Ψ0))[0]
#         w = solution_ODE_shooting.Ψ[0]

#         # plt.plot(r / angstrom, w)

#         if np.sign(w[-1]) == sign_lower:
#             # print("if statement")
#             E_lower = E_mid
#         else:
#             # print("else statement")
#             E_upper = E_mid

#     print("Hello, this is the end of the while loop")




############################### TEST Runge-Kutta ##################################
# psi = np.zeros_like(xs, dtype = "complex_")
# for i in range(len(xs)):
#     psi[i] = Runge_Kutta(x, delta_x, E, ϵ, Ψ0)[0]
#     # print(psi[i])
#     x += delta_x
# print(f"\n{psi}\n")