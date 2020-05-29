import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
import time

import pybamm
start_time = time.time()


I=-1
# constants
# Set Parameters values normally
# ------------------- Section w all input parameters (incl current)-------------------
R = 8.3145

F = 9.649 * (10**4)
v = 0.0114

EH0 = 2.35
EL0 = 2.18

k_p = 100
if I < 0:  # on charge, allow shuttle
    k_s0 = 0.00003  # at reference temperature T0, paper: 0.00003
else:
    k_s0 = 0.000000000000000001

temp_effect='off'

f_s = 0 # 0.25
S_star = 0.00005
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
V_min = 2.15
V_max = 2.5

T0 = 298  # K. reference temperature.
Ta = 298  # K. ambient temperature.
mc = 40  # cell mass in g TODO: should be a function of m_s
ch = 2  # J/(g K). cell specific heat capacity
h = 0.002  # W/k. total cell heat transfer coefficient  TODO: should be a function of mc or m_s
Na = 6.0221*10**(23)# 1/mol. Avogadro number
if temp_effect=='on':
    print('temperature effect on')

    A = 8.9712 * 10**(-20)  # J/mol. pre-exponential factor in Arrhenius eq for k_s
if temp_effect=='off':
    print('temperature effect off')
    A=0 # means ks has no temperature dependency and will always be k0.

else:
    print('temperature effect should be either on or off')
# ------------------- Section w useful groups of parameters -------------------
i_h_term_coef = ns8 * Ms * I * v * rho_s / (ne * F * k_p * (m_s**2))  # what is this used for?

i_coef = ne * F / (2 * R)
i_h_coef = -2 * ih0 * ar
i_l_coef = -2 * il0 * ar

E_H_coef = R / (4 * F)
f_h = (ns4**2) * Ms * v / ns8
f_l = (ns**2) * ns2 * Ms**2 * (v**2) / ns4

params = [R, F, v, EH0, EL0, k_p, k_s0, f_s, S_star,
          rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0,
          il0, ar, m_s, I, i_h_term_coef, i_coef,
          i_h_coef, i_l_coef, E_H_coef, f_h, f_l, T0, Ta, Na, h, A, ch,
          mc]  # create params as array w all parameters





def fT(S8, S4, S2, S, Sp, Ss, V, T, params):  # output is dTc/dt
    # unpack parameter list
    R, F, v, EH0, EL0, k_p, k_s0, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l, T0, Ta, Na, h, A, ch, mc = params

    return 1 / (mc * ch) * (fs(S8, S4, S2, S, Sp, Ss, V, T, params) * n4 * F / (ns8 * Ms) * V - h*(T - Ta))


def E_H_np(data, params):
    # unpack parameter list
    R, F, v, EH0, EL0, k_p, k_s0, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l, T0, Ta, Na, h, A, ch, mc = params

    # unpack data list
    S8, S4, S2, S, Sp, Ss, V, T = data

    return EH0 + E_H_coef * T * np.log(f_h * S8 / (S4**2))


def E_L_np(data, params):
    # unpack parameter list
    R, F, v, EH0, EL0, k_p, k_s0, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l, T0, Ta, Na, h, A, ch, mc = params

    # unpack data list
    S8, S4, S2, S, Sp, Ss, V, T = data

    return EL0 + E_H_coef * T * np.log(f_l * S4 / (S2 * (S**2)))


# Surface Overpotentials

def eta_H_np(data, params):
    # unpack data list
    S8, S4, S2, S, Sp, Ss, V, T = data

    return V - E_H_np(data, params)


def eta_L_np(data, params):
    # unpack data list
    S8, S4, S2, S, Sp, Ss, V, T = data

    return V - E_L_np(data, params)


# Half-cell Currents

def i_H_np(data, params):
    # unpack parameter list
    R, F, v, EH0, EL0, k_p, k_s0, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l, T0, Ta, Na, h, A, ch, mc = params

    S8, S4, S2, S, Sp, Ss, V, T = data

    return i_h_coef * np.sinh(i_coef / T * eta_H_np(data, params))


def i_L_np(data, params):
    # unpack parameter list
    R, F, v, EH0, EL0, k_p, k_s0, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l, T0, Ta, Na, h, A, ch, mc = params

    S8, S4, S2, S, Sp, Ss, V, T = data

    return i_l_coef * np.sinh(i_coef / T * eta_L_np(data, params))


def algebraic_condition_func_np(data, params):
    # unpack parameter list
    R, F, v, EH0, EL0, k_p, k_s0, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l, T0, Ta, Na, h, A, ch, mc = params

    return i_H_np(data, params) + i_L_np(data, params) - I


# -----------------------------RHS of ODE functions. Time evolution of all species. not used?
def f8_np(data, params):
    # unpack parameter list
    R, F, v, EH0, EL0, k_p, k_s0, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l, T0, Ta, Na, h, A, ch, mc = params

    # unpack data list
    S8, S4, S2, S, Sp, Ss, V, T = data
    k_s = k_s0 * np.exp(-A / R * Na * (1 / T - 1 / T0))
    return -(ns8 * Ms * i_H_np(data, params) / (n4 * F)) - k_s * S8


def f4_np(data, params):
    # unpack parameter list
    R, F, v, EH0, EL0, k_p, k_s0, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l, T0, Ta, Na, h, A, ch, mc = params

    # unpack data list
    S8, S4, S2, S, Sp, Ss, V, T = data
    k_s = k_s0 * np.exp(-A / R * Na * (1 / T - 1 / T0))
    return (2 * ns4 * Ms * i_H_np(data, params) / (n4 * F)) + (1 - (f_s * Ss / m_s)) * k_s * S8 - (
            ns4 * Ms * i_L_np(data, params) / (n4 * F))


def f2_np(data, params):
    # unpack parameter list
    R, F, v, EH0, EL0, k_p, k_s0, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l, T0, Ta, Na, h, A, ch, mc = params

    # unpack data list
    S8, S4, S2, S, Sp, Ss, V, T = data

    return ns2 * Ms * i_L_np(data, params) / (n4 * F)


def fp_np(data, params):
    # unpack parameter list
    R, F, v, EH0, EL0, k_p, k_s0, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l, T0, Ta, Na, h, A, ch, mc = params

    # unpack data list
    S8, S4, S2, S, Sp, Ss, V, T = data

    return k_p * Sp * (S - S_star) / (v * rho_s)


#
# def fs_np(data, params):
#     # unpack parameter list
#     R, F, v, EH0, EL0, k_p, k_s0, f_s, S_star, \
#     rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
#     il0, ar, m_s, I, i_h_term_coef, i_coef, \
#     i_h_coef, i_l_coef, E_H_coef, f_h, f_l,T0,Ta,Na,h,A,ch,mc = params
#
#     # unpack data list
#     S8, S4, S2, S, Sp, Ss, V, T = data
#     k_s=k_s0 * np.exp(-A / R * Na * (1 / T - 1 / T0))
#     return k_s * S8


def f_np(data, params):
    # unpack parameter list
    R, F, v, EH0, EL0, k_p, k_s0, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l, T0, Ta, Na, h, A, ch, mc = params

    # unpack data list
    S8, S4, S2, S, Sp, Ss, V, T = data

    return (2 * ns * Ms * i_L_np(data, params) / (n4 * F)) - (k_p * Sp * (S - S_star) / (v * rho_s))


################# Nested Functions using PyBaMM ##################################
# functions used to run simulation after initial cond have been established
# Nernst Potentials

def E_H(data, params):
    # unpack parameter list
    R, F, v, EH0, EL0, k_p, k_s0, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l, T0, Ta, Na, h, A, ch, mc = params

    # unpack data list
    S8, S4, S2, S, Sp, Ss, V, T = data

    return EH0 + E_H_coef * T * pybamm.log(f_h * S8 / (S4**2))


def E_L(data, params):
    # unpack parameter list
    R, F, v, EH0, EL0, k_p, k_s0, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l, T0, Ta, Na, h, A, ch, mc = params

    # unpack data list
    S8, S4, S2, S, Sp, Ss, V, T = data

    return EL0 + E_H_coef * T * pybamm.log(f_l * S4 / (S2 * (S**2)))


# Surface Overpotentials

def eta_H(data, params):
    # unpack data list
    S8, S4, S2, S, Sp, Ss, V, T = data

    return V - E_H(data, params)


def eta_L(data, params):
    # unpack data list
    S8, S4, S2, S, Sp, Ss, V, T = data

    return V - E_L(data, params)


# Half-cell Currents

def i_H(data, params):
    # unpack parameter list
    R, F, v, EH0, EL0, k_p, k_s0, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l, T0, Ta, Na, h, A, ch, mc = params

    S8, S4, S2, S, Sp, Ss, V, T = data

    return i_h_coef * pybamm.sinh(i_coef / T * eta_H(data, params))


def i_L(data, params):
    # unpack parameter list
    R, F, v, EH0, EL0, k_p, k_s0, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l, T0, Ta, Na, h, A, ch, mc = params

    S8, S4, S2, S, Sp, Ss, V, T = data

    return i_l_coef * pybamm.sinh(i_coef / T * eta_L(data, params))


def algebraic_condition_func(S8, S4, S2, S, Sp, Ss, V, T, params):
    # unpack parameter list
    R, F, v, EH0, EL0, k_p, k_s0, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l, T0, Ta, Na, h, A, ch, mc = params
    # pack data list
    data = S8, S4, S2, S, Sp, Ss, V, T

    # return abs(iH) + abs(iL) - I
    return i_H(data, params) + i_L(data, params) - I


################### Dynamic Equations ########################################

# RHS of ODE functions
def f8(S8, S4, S2, S, Sp, Ss, V, T, params):
    # unpack parameter list
    R, F, v, EH0, EL0, k_p, k_s0, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l, T0, Ta, Na, h, A, ch, mc = params

    # pack data list
    data = S8, S4, S2, S, Sp, Ss, V, T

    k_s = k_s0 * pybamm.exp(-A / R * Na * (1 / T - 1 / T0))
    return -(ns8 * Ms * i_H(data, params) / (n4 * F)) - k_s * S8


def f4(S8, S4, S2, S, Sp, Ss, V, T, params):
    # unpack parameter list
    R, F, v, EH0, EL0, k_p, k_s0, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l, T0, Ta, Na, h, A, ch, mc = params

    # pack data list
    data = S8, S4, S2, S, Sp, Ss, V, T
    k_s = k_s0 * pybamm.exp(-A / R * Na * (1 / T - 1 / T0))

    return (2 * ns4 * Ms * i_H(data, params) / (n4 * F)) + (1 - (f_s * Ss / m_s)) * k_s * S8 - (
            ns4 * Ms * i_L(data, params) / (n4 * F))


def f2(S8, S4, S2, S, Sp, Ss, V, T, params):
    # unpack parameter list
    R, F, v, EH0, EL0, k_p, k_s0, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l, T0, Ta, Na, h, A, ch, mc = params

    # pack data list
    data = S8, S4, S2, S, Sp, Ss, V, T

    return ns2 * Ms * i_L(data, params) / (n4 * F)


def fp(S8, S4, S2, S, Sp, Ss, V, T, params):
    # unpack parameter list
    R, F, v, EH0, EL0, k_p, k_s0, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l, T0, Ta, Na, h, A, ch, mc = params

    return k_p * Sp * (S - S_star) / (v * rho_s)


def fs(S8, S4, S2, S, Sp, Ss, V, T, params):
    # unpack parameter list
    R, F, v, EH0, EL0, k_p, k_s0, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l, T0, Ta, Na, h, A, ch, mc = params


    k_s = k_s0 * pybamm.exp(-A / R * Na * (1 / T - 1 / T0))

    return k_s * S8


def f(S8, S4, S2, S, Sp, Ss, V, T, params):
    # unpack parameter list
    R, F, v, EH0, EL0, k_p, k_s0, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l, T0, Ta, Na, h, A, ch, mc = params

    # pack data list
    data = S8, S4, S2, S, Sp, Ss, V, T

    return (2 * ns * Ms * i_L(data, params) / (n4 * F)) - (k_p * Sp * (S - S_star) / (v * rho_s))

    # ----------------------- Initial conditions -----------------------

def fl(S8, S4, S2, S, Sp, Ss, V, T, params):
    # unpack parameter list
    R, F, v, EH0, EL0, k_p, k_s0, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l, T0, Ta, Na, h, A, ch, mc = params

    # pack data list
    data = S8, S4, S2, S, Sp, Ss, V, T

    k_s = k_s0 * pybamm.exp(-A / R * Na * (1 / T - 1 / T0))
    return ((f_s * Ss / m_s)) * k_s * S8
if I>0:
    S8_initial = 0.998 * m_s
    S4_initial = 0.001 * m_s
    # S_initial = S_star # not defined in matlab, instead initial Sp is defined as:
    Sp_initial = 0.000001 * m_s  # in matlab
    Ss_initial = 0
else: # from 1A discharge
    S8_initial = 7.40E-14 #TODO: should be defined as share of m_s

    S4_initial = 1.56E-3 #TODO: should be defined as share of m_s
    # S_initial = S_star # not defined in matlab, instead initial Sp is defined as:
    Sp_initial = 0.5*m_s
    Ss_initial = 0


CellQ = m_s * 12 / 8 * F / Ms * 1 / 3600   # cell capacity in Ah. used to calculate approx discharge/charge duration

    ########################## Derived Initial Conditions ##################################

# Solve for initial voltage
T_initial = Ta  # initial cell temperature assumed to be ambient temperature
# create temp data array for initial conditions:
data_temp = (S8_initial, S4_initial, 'null', 'null', Sp_initial, 'null', 'null',T_initial)

V_initial = E_H_np(data_temp, params)  # zero overpotential
EL_initial = E_H_np(data_temp, params)  # zero overpotential

# How much S and S2 have been produced to satisfy EL=EH.  not with respect to mass ratio but w respect to simplified activity (stoich raised to power)
Sprod = f_l * S4_initial / np.exp((EL_initial - EL0) * n4 * F / (R * T_initial))  # 0.655=f_l in matlab

x0 = Sprod**(1 / float(3))
  # The starting estimate for the roots of fS_in, assuming the mass of the two species are similar. satisfy EL eq

def fS_in(x, m_s, S8_initial, S4_initial, Sp_initial,
          Sprod):  # identifying S and S2 mass that fullfil El, and mass conservation
    # substituting S2 by Sprod/S^2 from El eq and using mass conservation eq.
    return (x**2 * (m_s - S8_initial - S4_initial - Sp_initial - x) - Sprod)


S_initial = fsolve(fS_in, x0, args=(m_s, S8_initial, S4_initial, Sp_initial, Sprod))[0]
print(CellQ,'q')

S2_initial = m_s - S8_initial - S4_initial - Sp_initial - S_initial
print(S_initial, S2_initial)
# we now have the initial conditions before equilibrating the cell at zero current
# test if these satisfy the following:
# test_data = (S8_initial, S4_initial, S2_initial, S_initial, Sp_initial, Ss_initial, V_initial)
# error_mass = m_s - S8_initial - S4_initial - S2_initial - S_initial - Sp_initial
# error_iH = i_H_np(test_data, params)
# error_iL = i_L_np(test_data, params)
# error_etaH = 0 - eta_H_np(test_data, params)
# error_etaL = 0 - eta_L_np(test_data, params)
# error_EH = V_initial - E_H_np(test_data, params)
# print('errors before equilibration', error_iH, error_mass, error_iL, error_EH, error_etaH, error_etaL)
# print('concentrations + Voltage before equilibration', S8_initial, S4_initial, S2_initial, S_initial, Sp_initial,
#       V_initial)


# empty model framework
model = pybamm.BaseModel()

# ------------------------ Model Variables ------------------------
S8 = pybamm.Variable("S8")
S4 = pybamm.Variable("S4")
S2 = pybamm.Variable("S2")
S = pybamm.Variable("S")
Sp = pybamm.Variable("Sp")
Ss = pybamm.Variable("Ss")
V = pybamm.Variable("V")
T = pybamm.Variable("T")

model.variables = {"S8": S8, "S4": S4, "S2": S2, "S": S, "Sp": Sp, "Ss": Ss, "V": V, "T": T}

model.initial_conditions = {S8: pybamm.Scalar(S8_initial), S4: pybamm.Scalar(S4_initial),
                            S2: pybamm.Scalar(S2_initial),
                            S: pybamm.Scalar(S_initial), Sp: pybamm.Scalar(Sp_initial),
                            Ss: pybamm.Scalar(Ss_initial),
                            V: pybamm.Scalar(V_initial), T: pybamm.Scalar(T_initial)}

# ODEs
dS8dt = f8(S8, S4, S2, S, Sp, Ss, V, T, params)
dS4dt = f4(S8, S4, S2, S, Sp, Ss, V, T, params)
dS2dt = f2(S8, S4, S2, S, Sp, Ss, V, T, params)
dSpdt = fp(S8, S4, S2, S, Sp, Ss, V, T, params)
dSsdt = fs(S8, S4, S2, S, Sp, Ss, V, T, params)
dSdt = f(S8, S4, S2, S, Sp, Ss, V, T, params)
dTdt = fT(S8, S4, S2, S, Sp, Ss, V, T, params)

# Algebraic Condition
algebraic_condition = algebraic_condition_func(S8, S4, S2, S, Sp, Ss, V, T, params)

############# Model implementation ###############################################
model.rhs = {S8: dS8dt,
             S4: dS4dt,
             S2: dS2dt,
             S: dSdt,
             Sp: dSpdt,
             Ss: dSsdt,
             T: dTdt}
model.algebraic = {V: algebraic_condition}

CellQ = m_s * 12 / 8 * F / Ms * 1 / 3600  # cell capacity in Ah. used to calculate approx discharge/charge duration

if I > 0:
    model.events = {pybamm.Event("Maximum voltage", V_max - V),
                    pybamm.Event("Sp", Sp - 0.49869 * m_s)}  # Events will stop the solver whenever they return 0
    seconds = abs(CellQ / I * 3600)

if I < 0:
    model.events = {pybamm.Event("Minimum voltage", V - V_min),
                    pybamm.Event("S4", S4 - 1e-6)}  # Events will stop the solver whenever they return 0
    seconds = abs(CellQ / I * 3600*1.5)  # allow for longer charge times due to shuttle during slow discharge
disc = pybamm.Discretisation()  # use the default discretisation
disc.process_model(model);

# tol = 1e-9
# dae_solver = pybamm.CasadiSolver(mode="safe",
#                                  atol=tol,
#                                  rtol=tol,
#                                  root_tol=tol,
#                                  max_step_decrease_count=15)
dae_solver = pybamm.ScikitsDaeSolver(atol=1e-16, rtol=1e-9,max_steps=2000)  # TODO: decide on solver or test if both can be used

dt = 0.5 # has to very small if starting voltage is very different from voltage under load ~0.0015
t = np.linspace(0, seconds, int(seconds / dt))
solution = dae_solver.solve(model, t)

# retrieve solution data
t_sol = solution.t
S8_sol = solution["S8"].data
S4_sol = solution["S4"].data
S2_sol = solution["S2"].data
S_sol = solution["S"].data
Sp_sol = solution["Sp"].data
Ss_sol = solution["Ss"].data
V_sol = solution["V"].data
T_sol = solution["T"].data
solution = t_sol, S8_sol, S4_sol, S2_sol, S_sol, Sp_sol, Ss_sol, V_sol, T_sol



# Note that none of these plot functions require the current,
# so using the last params is perfectly ok



# repack species only data
species_data = S8_sol, S4_sol, S2_sol, S_sol, Sp_sol, Ss_sol, V_sol, T_sol
# derive solution data
E_H_sol = E_H_np(species_data, params)
E_L_sol = E_L_np(species_data, params)
eta_H_sol = eta_H_np(species_data, params)
eta_L_sol = eta_L_np(species_data, params)
i_H_sol = i_H_np(species_data, params)
i_L_sol = i_L_np(species_data, params)

# plotting

plt.figure(1)
plt.plot(t_sol, V_sol)
plt.xlabel('time [s]')
plt.ylabel('Voltage [V]')

plt.figure(2)
plt.plot(t_sol, S8_sol)
plt.plot(t_sol, S4_sol)
plt.plot(t_sol, S2_sol)
plt.plot(t_sol, S_sol)
plt.plot(t_sol, Ss_sol)
plt.plot(t_sol, Sp_sol)
plt.xlabel('time [s]')
plt.ylabel('Species [g]')

plt.legend(['$S_8$', '$S_4$','$S_2$', '$S$', '$S_s$', '$S_p$'])
# plt.legend([ '$S$', '$S_s$', '$S_p$'])

plt.figure(4)
plt.plot(t_sol, E_H_sol)
plt.plot(t_sol, E_L_sol)
plt.xlabel('time [s]')
plt.ylabel('Potential')
plt.legend(['$E_H$', '$E_L$'])
plt.title('Cell Potentials')

plt.figure(5)
plt.plot(t_sol, eta_H_sol)
plt.plot(t_sol, eta_L_sol)
plt.xlabel('time [s]')
plt.ylabel('Over-Potential')
plt.legend(['$\eta_H$', '$\eta_L$'])
plt.title('Cell Over-Potentials')

plt.figure(6)
plt.plot(t_sol, i_H_sol)
plt.plot(t_sol, i_L_sol)
plt.xlabel('time [s]')
plt.ylabel('Current [A]')
plt.legend(['$i_H$', '$i_L$'])
plt.title('Cell Currents')

plt.figure(3)
plt.plot(t_sol, T_sol)

plt.xlabel('time [s]')
plt.ylabel('Temperature [K]')
plt.legend(['Cell Temperature'])
plt.title('Cell Temperature')

plt.show()
print("--- %s seconds ---" % (time.time() - start_time))