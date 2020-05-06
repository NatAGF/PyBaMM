import matplotlib.pyplot as plt
import numpy as np

import pybamm

# empty model framework
model = pybamm.BaseModel()

#
#
# # constants
# Na=pybamm.Parameter("mol^-1. Avogadro No")
# R=pybamm.Parameter("J K^-1 mol^-1 universal gas constant")
# F=pybamm.Parameter("C mol-1 Faraday constant")
#
# # cell design parameters
# Ve=pybamm.Parameter("L. Volume of electrolyte per cell")
# mS=pybamm.Parameter("g. Mass of active sulfur per cell")
# ar=pybamm.Parameter("active reaction area per cell [m2]")
#
#
# # sulfur properties
# mm=pybamm.Parameter("Molar mass of S [g/mole]")
# ne=pybamm.Parameter("#. No of electrons transferred in electrochemical reactions")
# rhoS=pybamm.Parameter("g/L. S density")
# nS=pybamm.Parameter("number of S molecules in S")
# nS2=pybamm.Parameter("number of S molecules in S2")
# nS4=pybamm.Parameter("number of S molecules in S4")
# nS8=pybamm.Parameter("number of S molecules in S8")
#
# # kinetics and other modelling parameters
# EH0=pybamm.Parameter("Ref potential for high plateau electrochemical reaction [V]")
# EL0=pybamm.Parameter("V. Ref potential for low plateau electrochemical reaction")
# iH0=pybamm.Parameter("A/m^2. Exchange current density")
# iL0=pybamm.Parameter("A/m^2. Exchange current density")
#
#
# T=pybamm.Parameter("K. Temperature")

# constants
# Set Parameters values normally

R = 8.3145
T = 298
F = 9.649 * (10**4)
v = 0.0114

EH0 = 2.35
EL0 = 2.18

k_p = 100
k_s = 0  # 0.0002 for charge
f_s = 0.25
S_star = 0.0001
rho_s = 2 * (10**3)

Ms = 32
ne = 4
ns = 1
ns2 = 2
ns4 = 4
ns8 = 8
n4 = 4

ih0 = 1
il0 = 0.5
ar = 0.960
m_s = 2.7
I = 0.08  # change

i_h_term_coef = ns8 * Ms * I * v * rho_s / (ne * F * k_p * (m_s**2))

i_coef = ne * F / (2 * R * T)
i_h_coef = -2 * ih0 * ar
i_l_coef = -2 * il0 * ar

E_H_coef = R * T / (4 * F)
f_h = (ns4**2) * Ms * v / ns8
f_l = (ns**2) * ns2 * Ms**2 * (v**2) / ns4
V_min=2.1
V_max=2.45
params = [R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star,
          rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0,
          il0, ar, m_s, I, i_h_term_coef, i_coef,
          i_h_coef, i_l_coef, E_H_coef, f_h, f_l]

# ------------------------ Variables ------------------------
S8 = pybamm.Variable("S8")
S4 = pybamm.Variable("S4")
S2 = pybamm.Variable("S2")
S = pybamm.Variable("S")
Sp = pybamm.Variable("Sp")
Ss = pybamm.Variable("Ss")
V = pybamm.Variable("V")


model.variables = {"S8": S8, "S4": S4, "S2": S2, "S": S, "Sp": Sp, "Ss": Ss, "V": V}


# ----------------------- Governing equations using numpy -----------------------
# Nernst linking overpotential and species concentrations


def E_H_np(data, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    # unpack data list
    S8, S4, S2, S, Sp, Ss, V = data

    return EH0 + E_H_coef * np.log(f_h * S8 / (S4**2))


def E_L_np(data, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    # unpack data list
    S8, S4, S2, S, Sp, Ss, V = data

    return EL0 + E_H_coef * np.log(f_l * S4 / (S2 * (S**2)))


# Surface Overpotentials

def eta_H_np(data, params):
    # unpack data list
    S8, S4, S2, S, Sp, Ss, V = data

    return V - E_H_np(data, params)


def eta_L_np(data, params):
    # unpack data list
    S8, S4, S2, S, Sp, Ss, V = data

    return V - E_L_np(data, params)


# Half-cell Currents

def i_H_np(data, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    return i_h_coef * np.sinh(i_coef * eta_H_np(data, params))


def i_L_np(data, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    return i_l_coef * np.sinh(i_coef * eta_L_np(data, params))


def algebraic_condition_func_np(data, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    return i_H_np(data, params) + i_L_np(data, params) - I


# RHS of ODE functions
def f8_np(data, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    # unpack data list
    S8, S4, S2, S, Sp, Ss, V = data

    return -(ns8 * Ms * i_H_np(data, params) / (n4 * F)) - k_s * S8


def f4_np(data, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    # unpack data list
    S8, S4, S2, S, Sp, Ss, V = data

    return (2 * ns4 * Ms * i_H_np(data, params) / (n4 * F)) + (1 - (f_s * Ss / m_s)) * k_s * S8 - (
            ns4 * Ms * i_L_np(data, params) / (n4 * F))


def f2_np(data, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    # unpack data list
    S8, S4, S2, S, Sp, Ss, V = data

    return ns2 * Ms * i_L_np(data, params) / (n4 * F)


def fp_np(data, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    # unpack data list
    S8, S4, S2, S, Sp, Ss, V = data

    return k_p * Sp * (S - S_star) / (v * rho_s)


def fs_np(data, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    # unpack data list
    S8, S4, S2, S, Sp, Ss, V = data

    return k_s * S8


def f_np(data, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    # unpack data list
    S8, S4, S2, S, Sp, Ss, V = data

    return (2 * ns * Ms * i_L_np(data, params) / (n4 * F)) - (k_p * Sp * (S - S_star) / (v * rho_s))


# # time derivatives of species concentration
#
# dS8dt = -(nS8 * mm / (ne * F) * iH(iH0, ar, ne, F, V, S8, S4, EH0, fH)) - ks * S8
# dS4dt = 2 * (nS4 * mm) / (ne * F) * iH(iH0, ar, ne, F, V, S8, S4, EH0, fH) + (1 - (fs * Ss / mm)) * ks * S8 - (
#             nS4 * mm / ne * F) * iL(iL0, ar, ne, F, V, S4, S2, S, EL0, fL)
# dS2dt = nS2 * mm * iL(iL0, ar, ne, F, V, S4, S2, S, EL0, fL) / (ne * F)
# dSdt = (2 * nS * mm) / (ne * F) * iL(iL0, ar, ne, F, V, S4, S2, S, EL0, fL) - (kp * Sp * (S - Ssat) / (Ve * rhoS))
# dSpdt = 1 / (Ve * rhoS) * kp * Sp * (S - Ssat)
# dSsdt = ks * S8
S8_initial = 0.998 * m_s
S4_initial = 0.001 * m_s
# S_initial = S_star
S_initial=0.0000001*m_s
Ss_initial = 0
I_initial = 0

# ----------------------- Initial conditions approach I------------------
# Solve for initial voltage

########################## Derived Initial Conditions ##################################

# Solve for initial voltage
data1 = S8_initial, S4_initial, S_initial, 'null', 'null', Ss_initial, 'null'
V_initial = E_H_np(data1, params)

# Solve for S2_initial
S2_initial = np.exp(n4 * F * (EL0 - V_initial) / (R * T)) * (f_l * S4_initial / (S_initial**2))

# Solve for Sp_initial
Sp_initial = m_s - S8_initial - S4_initial - S2_initial - S_initial - Ss_initial
data1 = S8_initial, S4_initial, S_initial, 'null', 'null', Ss_initial, 'null'
V_initial = E_H_np(data1, params)

# Initial Conditions
model.initial_conditions = {S8: pybamm.Scalar(S8_initial), S4: pybamm.Scalar(S4_initial), S2: pybamm.Scalar(S2_initial),
                            S: pybamm.Scalar(S_initial), Sp: pybamm.Scalar(Sp_initial), Ss: pybamm.Scalar(Ss_initial),
                            V: pybamm.Scalar(V_initial)}


################# Nested Functions With PyBaMM ##################################

# Nernst Potentials

def E_H(data, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    # unpack data list
    S8, S4, S2, S, Sp, Ss, V = data

    return EH0 + E_H_coef * pybamm.log(f_h * S8 / (S4**2))


def E_L(data, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    # unpack data list
    S8, S4, S2, S, Sp, Ss, V = data

    return EL0 + E_H_coef * pybamm.log(f_l * S4 / (S2 * (S**2)))


# Surface Overpotentials

def eta_H(data, params):
    # unpack data list
    S8, S4, S2, S, Sp, Ss, V = data

    return V - E_H(data, params)


def eta_L(data, params):
    # unpack data list
    S8, S4, S2, S, Sp, Ss, V = data

    return V - E_L(data, params)


# Half-cell Currents

def i_H(data, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    return i_h_coef * pybamm.sinh(i_coef * eta_H(data, params))


def i_L(data, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    return i_l_coef * pybamm.sinh(i_coef * eta_L(data, params))


def algebraic_condition_func(S8, S4, S2, S, Sp, Ss, V, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    # pack data list
    data = S8, S4, S2, S, Sp, Ss, V

    return i_H(data, params) + i_L(data, params) - I


################### Dynamic Equations ########################################

# RHS of ODE functions
def f8(S8, S4, S2, S, Sp, Ss, V, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    # pack data list
    data = S8, S4, S2, S, Sp, Ss, V

    return -(ns8 * Ms * i_H(data, params) / (n4 * F)) - k_s * S8


def f4(S8, S4, S2, S, Sp, Ss, V, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    # pack data list
    data = S8, S4, S2, S, Sp, Ss, V

    return (2 * ns4 * Ms * i_H(data, params) / (n4 * F)) + (1 - (f_s * Ss / m_s)) * k_s * S8 - (
            ns4 * Ms * i_L(data, params) / (n4 * F))


def f2(S8, S4, S2, S, Sp, Ss, V, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    # pack data list
    data = S8, S4, S2, S, Sp, Ss, V

    return ns2 * Ms * i_L(data, params) / n4 * F


def fp(S8, S4, S2, S, Sp, Ss, V, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    return k_p * Sp * (S - S_star) / (v * rho_s)


def fs(S8, S4, S2, S, Sp, Ss, V, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    return k_s * S8


def f(S8, S4, S2, S, Sp, Ss, V, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    # pack data list
    data = S8, S4, S2, S, Sp, Ss, V

    return (2 * ns * Ms * i_L(data, params) / (n4 * F)) - (k_p * Sp * (S - S_star) / (v * rho_s))


# ODEs
dS8dt = f8(S8, S4, S2, S, Sp, Ss, V, params)
dS4dt = f4(S8, S4, S2, S, Sp, Ss, V, params)
dS2dt = f2(S8, S4, S2, S, Sp, Ss, V, params)
dSpdt = fp(S8, S4, S2, S, Sp, Ss, V, params)
dSsdt = fs(S8, S4, S2, S, Sp, Ss, V, params)
dSdt = f(S8, S4, S2, S, Sp, Ss, V, params)


# Algebraic Condition
algebraic_condition = algebraic_condition_func(S8, S4, S2, S, Sp, Ss, V, params)

############# Model implementation ###############################################
model.rhs = {S8: dS8dt,
             S4: dS4dt,
             S2: dS2dt,
             S: dSdt,
             Sp: dSpdt,
             Ss: dSsdt,}
model.algebraic = {V: algebraic_condition}

model.events = {pybamm.Event("Minimum voltage", V - V_min),pybamm.Event("Maximum voltage", V_max-V)} #Events will stop the solver whenever they return 0



disc = pybamm.Discretisation()  # use the default discretisation
disc.process_model(model);

# solver initiated
dae_solver = pybamm.ScikitsDaeSolver(atol=1e-2, rtol=1e-2)


seconds = 3600 * 25
dt = 1 # change, 0,2 work for I=3, 2 works for I=1
t = np.linspace(0, seconds, int(seconds / dt))
solution = dae_solver.solve(model, t)

# retrieve data
t_sol = solution.t
S8_sol = solution["S8"].data
S4_sol = solution["S4"].data
S2_sol = solution["S2"].data
S_sol = solution["S"].data
Sp_sol = solution["Sp"].data
Ss_sol = solution["Ss"].data
V_sol = solution["V"].data
# V_sol = solution["V"].data


data = t_sol, S8_sol, S4_sol, S2_sol, S_sol, Sp_sol, Ss_sol, V_sol

plt.figure(1)
plt.plot(t_sol, V_sol)
plt.xlabel('time [s]')
plt.ylabel('Voltage [V]')

plt.figure(3)
plt.plot(t_sol, S8_sol)
plt.plot(t_sol, S4_sol)
plt.plot(t_sol, S_sol)
plt.plot(t_sol, Ss_sol)
plt.plot(t_sol, Sp_sol)
plt.xlabel('time [s]')
plt.ylabel('Species [g]')
plt.legend(['$S_8$', '$S_4$','$S$', '$Ss$', '$S_p$',])

plt.show()


def norm(x):
    return np.dot(x, x) / len(x)


def algebraic_test(data, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    # unpack data list
    t_sol, S8_sol, S4_sol, S2_sol, S_sol, Sp_sol, Ss_sol, V_sol = data

    # algebraic condition
    I_list = np.zeros(len(V_sol))
    for i in range(len(V_sol)):
        temp_data = S8_sol[i], S4_sol[i], S2_sol[i], S_sol[i], Sp_sol[i], Ss_sol[i], V_sol[i]
        I_list[i] = algebraic_condition_func_np(temp_data, params)

    return norm(I_list)


def numerical_deriv(x, t):
    n = len(x)
    dt = t[1] - t[0]
    return (x[1:n] - x[0:n - 1]) / dt


def derivative_test(data, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    # unpack data list
    t_sol, S8_sol, S4_sol, S2_sol, S_sol, Sp_sol, Ss_sol, V_sol = data

    # repack data
    n = len(t_sol)
    data = S8_sol[1:n], S4_sol[1:n], S2_sol[1:n], S_sol[1:n], Sp_sol[1:n], Ss_sol[1:n], V_sol[1:n]

    # evaluate norms
    S8_norm = norm(numerical_deriv(S8_sol, t_sol) - f8_np(data, params))
    S4_norm = norm(numerical_deriv(S4_sol, t_sol) - f4_np(data, params))
    S2_norm = norm(numerical_deriv(S2_sol, t_sol) - f2_np(data, params))
    Sp_norm = norm(numerical_deriv(Sp_sol, t_sol) - fp_np(data, params))
    Ss_norm = norm(numerical_deriv(Ss_sol, t_sol) - fs_np(data, params))
    S_norm = norm(numerical_deriv(S_sol, t_sol) - f_np(data, params))

    return S8_norm, S4_norm, S2_norm, S_norm, Sp_norm, Ss_norm


def psuedo_mass_conservation_test(data, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    # unpack data list
    t_sol, S8_sol, S4_sol, S2_sol, S_sol, Sp_sol, Ss_sol, V_sol = data

    # repack data
    n = len(t_sol)
    data = S8_sol[1:n], S4_sol[1:n], S2_sol[1:n], S_sol[1:n], Sp_sol[1:n], Ss_sol[1:n], V_sol[1:n]

    # evaluate norms
    Sum_data = f8_np(data, params) + f4_np(data, params) \
               + f2_np(data, params) + fp_np(data, params) \
               + fs_np(data, params) + f_np(data, params) \
               - (1 - (f_s * Ss_sol[1:n] / m_s)) * k_s * S8_sol[1:n]

    return norm(Sum_data)
print('Algebraic Error: {}'.format(algebraic_test(data,params)))
print('Maximum Derivative Error: {}'.format(np.max(derivative_test(data,params))))
print('Psuedo Mass Conservation Error: {}'.format(psuedo_mass_conservation_test(data,params)))

# I=0.01
ramping='off'
# initialize parameters
if ramping=='on':
# ramp up over one second
    dtr = 1e-2
    n = int(1 / dtr)
    t_ramp = np.linspace(0, 1, n)

    # create space of currents from zero to initial
    I_initial = I  # desired initial condition
    I_ramp = np.linspace(0, I_initial, n)

    t_temp = np.linspace(0, dt, 2)

    I = I_ramp[0]
    params = [R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star,
              rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0,
              il0, ar, m_s, I, i_h_term_coef, i_coef,
              i_h_coef, i_l_coef, E_H_coef, f_h, f_l]

    # use previous zero current initial conditions
    model.initial_conditions = {S8: pybamm.Scalar(S8_initial),
                                S4: pybamm.Scalar(S4_initial),
                                S2: pybamm.Scalar(S2_initial),
                                S: pybamm.Scalar(S_initial),
                                Sp: pybamm.Scalar(Sp_initial),
                                Ss: pybamm.Scalar(Ss_initial),
                                V: pybamm.Scalar(V_initial)}

    # create time space for the single step
    t_temp = np.linspace(0, dt, 3)

    # store data
    S8_data = [S8_initial]
    S4_data = [S4_initial]
    S2_data = [S2_initial]
    S_data = [S_initial]
    Ss_data = [Ss_initial]
    Sp_data = [Sp_initial]
    V_data = [V_initial]

    for i in range(0, n):
        # define the solver
        model = pybamm.BaseModel()

        S8 = pybamm.Variable("S8")
        S4 = pybamm.Variable("S4")
        S2 = pybamm.Variable("S2")
        S = pybamm.Variable("S")
        Sp = pybamm.Variable("Sp")
        Ss = pybamm.Variable("Ss")
        V = pybamm.Variable("V")

        model.variables = {"S8": S8,
                           "S4": S4,
                           "S2": S2,
                           "S": S,
                           "Sp": Sp,
                           "Ss": Ss,
                           "V": V}

        model.rhs = {S8: dS8dt,
                     S4: dS4dt,
                     S2: dS2dt,
                     S: dSdt,
                     Sp: dSpdt,
                     Ss: dSsdt}

        model.algebraic = {V: algebraic_condition}
        model.events = {pybamm.Event("Minimum voltage", V - V_min),
                        pybamm.Event("Maximum voltage", V_max - V)}  # Events will stop the solver whenever they return 0

        # update current
        I = I_ramp[i]
        params = [R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star,
                  rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0,
                  il0, ar, m_s, I, i_h_term_coef, i_coef,
                  i_h_coef, i_l_coef, E_H_coef, f_h, f_l]

        # update new initial condition
        model.initial_conditions = {S8: pybamm.Scalar(S8_data[-1]),
                                    S4: pybamm.Scalar(S4_data[-1]),
                                    S2: pybamm.Scalar(S2_data[-1]),
                                    S: pybamm.Scalar(S_data[-1]),
                                    Sp: pybamm.Scalar(Sp_data[-1]),
                                    Ss: pybamm.Scalar(Ss_data[-1]),
                                    V: pybamm.Scalar(V_data[-1])}

        disc = pybamm.Discretisation()  # use the default discretisation
        disc.process_model(model)
        solution = dae_solver.solve(model, t_temp)

        # retrieve solution
        S8_sol = solution["S8"].data
        S4_sol = solution["S4"].data
        S2_sol = solution["S2"].data
        S_sol = solution["S"].data
        Sp_sol = solution["Sp"].data
        Ss_sol = solution["Ss"].data
        V_sol = solution["V"].data

        # store data
        S8_data.append(S8_sol[-1])
        S4_data.append(S4_sol[-1])
        S2_data.append(S2_sol[-1])
        S_data.append(S_sol[-1])
        Ss_data.append(Ss_sol[-1])
        Sp_data.append(Sp_sol[-1])
        V_data.append(V_sol[-1])

    plt.plot(V_data)

    model = pybamm.BaseModel()

    S8 = pybamm.Variable("S8")
    S4 = pybamm.Variable("S4")
    S2 = pybamm.Variable("S2")
    S = pybamm.Variable("S")
    Sp = pybamm.Variable("Sp")
    Ss = pybamm.Variable("Ss")
    V = pybamm.Variable("V")

    model.variables = {"S8": S8,
                       "S4": S4,
                       "S2": S2,
                       "S": S,
                       "Sp": Sp,
                       "Ss": Ss,
                       "V": V}

    model.rhs = {S8: dS8dt,
                 S4: dS4dt,
                 S2: dS2dt,
                 S: dSdt,
                 Sp: dSpdt,
                 Ss: dSsdt}

    model.algebraic = {V: algebraic_condition}

    # update current
    I = I_initial
    params = [R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star,
              rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0,
              il0, ar, m_s, I, i_h_term_coef, i_coef,
              i_h_coef, i_l_coef, E_H_coef, f_h, f_l]

    # update new initial condition
    model.initial_conditions = {S8: pybamm.Scalar(S8_data[-1]),
                                S4: pybamm.Scalar(S4_data[-1]),
                                S2: pybamm.Scalar(S2_data[-1]),
                                S: pybamm.Scalar(S_data[-1]),
                                Sp: pybamm.Scalar(Sp_data[-1]),
                                Ss: pybamm.Scalar(Ss_data[-1]),
                                V: pybamm.Scalar(V_data[-1])}

    dae_solver = pybamm.ScikitsDaeSolver(atol=1e-6, rtol=1e-6)

    t = np.linspace(0, seconds, int(seconds / dt))
    disc = pybamm.Discretisation()  # use the default discretisation
    disc.process_model(model)
    solution = dae_solver.solve(model, t)

    # retrieve data
    t_sol  = solution.t
    S8_sol = solution["S8"].data
    S4_sol = solution["S4"].data
    S2_sol = solution["S2"].data
    S_sol  = solution["S"].data
    Sp_sol = solution["Sp"].data
    Ss_sol = solution["Ss"].data
    V_sol  = solution["V"].data

    data = t_sol, S8_sol, S4_sol, S2_sol, S_sol, Sp_sol, Ss_sol, V_sol

    # plotting


    plt.figure(1)
    plt.plot(t_sol, V_sol)
    plt.xlabel('time [s]')
    plt.ylabel('Voltage [V]')

    plt.figure(3)
    plt.plot(t_sol, S8_sol)
    plt.plot(t_sol, S4_sol)
    plt.xlabel('time [s]')
    plt.ylabel('Species [g]')
    plt.legend(['$S_8$', '$S_4$'])

    plt.figure(4)
    plt.plot(t_sol, S_sol)
    plt.plot(t_sol, Ss_sol)
    plt.plot(t_sol, Sp_sol)
    plt.xlabel('time [s]')
    plt.ylabel('Species [g]')
    plt.legend(['$S$', '$S_s$', '$S_p$']);

    plt.show()
