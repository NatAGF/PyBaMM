from cycle_input import cycle_func
import matplotlib.pyplot as plt
solution=cycle_func([1,-1,1])

t_sol = solution[0]
S8_sol = solution[1]
S4_sol = solution[2]
S2_sol = solution[3]
S_sol = solution[4]
Sp_sol = solution[5]
Ss_sol = solution[6]
V_sol = solution[7]
T_sol = solution[8]

E_H_sol = solution[9]
E_L_sol = solution[10]
eta_H_sol = solution[11]
eta_L_sol = solution[12]
i_H_sol = solution[13]
i_L_sol = solution[14]

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