import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
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
# ------------------- Section w all input parameters (incl current)-------------------
R = 8.3145
T = 298
F = 9.649 * (10**4)
v = 0.0114

EH0 = 2.35
EL0 = 2.18

k_p = 100
k_s = 0  # 0.00003 for charge
f_s = 0.25
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

I_op = 12.5 # user input
I_initial=0
I=I_op


# ------------------- Section w useful groups of parameters -------------------
i_h_term_coef = ns8 * Ms * I * v * rho_s / (ne * F * k_p * (m_s**2)) # what is this used for?

i_coef = ne * F / (2 * R * T)
i_h_coef = -2 * ih0 * ar
i_l_coef = -2 * il0 * ar

E_H_coef = R * T / (4 * F)
f_h = (ns4**2) * Ms * v / ns8
f_l = (ns**2) * ns2 * Ms**2 * (v**2) / ns4

params = [R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star,
          rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0,
          il0, ar, m_s, I, i_h_term_coef, i_coef,
          i_h_coef, i_l_coef, E_H_coef, f_h, f_l]  # create params as array w all parameters

# ------------------------ Model Variables ------------------------
S8 = pybamm.Variable("S8")
S4 = pybamm.Variable("S4")
S2 = pybamm.Variable("S2")
S = pybamm.Variable("S")
Sp = pybamm.Variable("Sp")
Ss = pybamm.Variable("Ss")
V = pybamm.Variable("V")

model.variables = {"S8": S8, "S4": S4, "S2": S2, "S": S, "Sp": Sp, "Ss": Ss, "V": V}


# ----------------------- Governing equations using numpy. Used to derive initial conditions -----------------------
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



# -----------------------------RHS of ODE functions. Time evolution of all species. not used?
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

#
# def fs_np(data, params):
#     # unpack parameter list
#     R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
#     rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
#     il0, ar, m_s, I, i_h_term_coef, i_coef, \
#     i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params
#
#     # unpack data list
#     S8, S4, S2, S, Sp, Ss, V = data
#
#     return k_s * S8


def f_np(data, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    # unpack data list
    S8, S4, S2, S, Sp, Ss, V = data


    return (2 * ns * Ms * i_L_np(data, params) / (n4 * F)) - (k_p * Sp * (S - S_star) / (v * rho_s))



################# Nested Functions using PyBaMM ##################################
# functions used to run simulation after initial cond have been established
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

    # return abs(iH) + abs(iL) - I
    return i_H(data, params)+i_L(data, params)-I

def algebraic_condition_func0(S8, S4, S2, S, Sp, Ss, V, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params
    # pack data list
    data = S8, S4, S2, S, Sp, Ss, V

    # return abs(iH) + abs(iL) - I
    return i_H(data, params)+i_L(data, params)-0


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

    return (2*ns4*Ms*i_H(data,params)/(n4*F)) + (1-(f_s*Ss/m_s))*k_s*S8 - (ns4*Ms*i_L(data,params)/(n4*F))


def f2(S8, S4, S2, S, Sp, Ss, V, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    # pack data list
    data = S8, S4, S2, S, Sp, Ss, V

    return ns2*Ms*i_L(data,params)/(n4*F)



def fp(S8, S4, S2, S, Sp, Ss, V, params):
    # unpack parameter list
    R, T, F, v, EH0, EL0, k_p, k_s, f_s, S_star, \
    rho_s, Ms, ne, ns, ns2, ns4, ns8, n4, ih0, \
    il0, ar, m_s, I, i_h_term_coef, i_coef, \
    i_h_coef, i_l_coef, E_H_coef, f_h, f_l = params

    return k_p*Sp*(S-S_star)/(v*rho_s)


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

    return (2*ns*Ms*i_L(data,params)/(n4*F)) - (k_p*Sp*(S-S_star)/(v*rho_s))


# ODEs
dS8dt = f8(S8, S4, S2, S, Sp, Ss, V, params)
dS4dt = f4(S8, S4, S2, S, Sp, Ss, V, params)
dS2dt = f2(S8, S4, S2, S, Sp, Ss, V, params)
dSpdt = fp(S8, S4, S2, S, Sp, Ss, V, params)
dSsdt = fs(S8, S4, S2, S, Sp, Ss, V, params)
dSdt = f(S8, S4, S2, S, Sp, Ss, V, params)

# ----------------------- Initial conditions -----------------------

S8_initial = 0.998 * m_s
S4_initial = 0.001 * m_s
# S_initial = S_star # not defined in matlab, instead initial Sp is defined as:
Sp_initial=0.000001*m_s #in matlab
Ss_initial = 0
CellQ=S8_initial*12/8*F/Ms*1/3600-1 #cell capacity in Ah. used to calculate approx discharge/charge duration

########################## Derived Initial Conditions ##################################

# Solve for initial voltage
data_temp = (S8_initial, S4_initial, 'null', 'null', Sp_initial, 'null','null') # create temp data array for initial conditions
V_initial = E_H_np(data_temp, params) # zero overpotential
EL_initial=E_H_np(data_temp, params) # zero overpotential

# How much S and S2 have been produced to satisfy EL=EH.  not with respect to mass ratio but w respect to simplified activity (stoich raised to power)
Sprod = f_l * S4_initial/np.exp((EL_initial-EL0)*n4*F/(R*T)) # 0.655=f_l in matlab

def fS_in(x,m_s,S8_initial,S4_initial,Sp_initial,Sprod): #identifying S and S2 mass that fullfil El, and mass conservation
    # substituting S2 by Sprod/S^2 from El eq and using mass conservation eq.
    return (x**2*(m_s-S8_initial-S4_initial-Sp_initial-x)-Sprod)
x0=Sprod**(1/float(3)) # The starting estimate for the roots of fS_in, assuming the mass of the two species are similar. satisfy EL eq

S_initial=fsolve(fS_in,x0,args=(m_s,S8_initial,S4_initial,Sp_initial,Sprod))[0] # change, should be S_initial

S2_initial=m_s - S8_initial - S4_initial - Sp_initial - S_initial
print((EL0 + E_H_coef * np.log(f_l * S4_initial / (S2_initial * (S_initial**2)))-V_initial), 'pybamm error')
print((EL0 + E_H_coef * np.log(f_l * S4_initial / (S_initial * (S2_initial**2)))-V_initial), 'matlab error')

# we now have the initial conditions before equilibrating the cell at zero current
# test if these satisfy the following:
test_data=(S8_initial, S4_initial,S2_initial,S_initial, Sp_initial,Ss_initial,V_initial)
error_mass=m_s - S8_initial - S4_initial - S2_initial - S_initial - Sp_initial
error_iH=i_H_np(test_data, params)
error_iL=i_L_np(test_data, params)
error_etaH=0-eta_H_np(test_data, params)
error_etaL=0-eta_L_np(test_data, params)
error_EH=V_initial-E_H_np(test_data, params)
print('errors before equilibration',error_iH,error_mass,error_iL,error_EH,error_etaH,error_etaL)
print('concentrations + Voltage before equilibration',S8_initial,S4_initial,S2_initial,S_initial,Sp_initial,V_initial)

# Initial Conditions used to equilibrate cell at zero current. these are "final"/equilibrated initial conditions if S<S_star.
model.initial_conditions = {S8: pybamm.Scalar(S8_initial), S4: pybamm.Scalar(S4_initial), S2: pybamm.Scalar(S2_initial),
                             S: pybamm.Scalar(S_initial), Sp: pybamm.Scalar(Sp_initial), Ss: pybamm.Scalar(Ss_initial),
                             V: pybamm.Scalar(V_initial)}
# Algebraic Condition
algebraic_condition = algebraic_condition_func(S8, S4, S2, S, Sp, Ss, V, params)

############# Model implementation ###############################################
model.rhs = {S8: dS8dt,
             S4: dS4dt,
             S2: dS2dt,
             S: dSdt,
             Sp: dSpdt,
             Ss: dSsdt, }
model.algebraic = {V: algebraic_condition}

model.events = {pybamm.Event("Minimum voltage", V - V_min),
                pybamm.Event("Maximum voltage", V_max - V)}  # Events will stop the solver whenever they return 0

disc = pybamm.Discretisation()  # use the default discretisation
disc.process_model(model);

# solver initiated

dae_solver = pybamm.ScikitsDaeSolver(atol=1e-14, rtol=1e-6)
seconds = CellQ/I_op*3600-1 # to terminate before solver experiences error. should be adjusted to include voltage drop off
print(seconds)
dt = 1
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


# plotting

plt.figure(1)
plt.plot(t_sol, V_sol)
plt.xlabel('time [s]')
plt.ylabel('Voltage [V]')

plt.figure(2)
plt.plot(t_sol, S8_sol)
plt.plot(t_sol, S4_sol)
plt.xlabel('time [s]')
plt.ylabel('Species [g]')
plt.legend(['$S_8$', '$S_4$'])

plt.figure(3)
plt.plot(t_sol, S2_sol)
plt.plot(t_sol, S_sol)
plt.plot(t_sol, Ss_sol)
plt.plot(t_sol, Sp_sol)
plt.xlabel('time [s]')
plt.ylabel('Species [g]')

plt.legend(['$S_2$', '$S$', '$S_s$', '$S_p$'])
# plt.legend([ '$S$', '$S_s$', '$S_p$'])

plt.show()
