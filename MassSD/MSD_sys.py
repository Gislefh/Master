#Mass Spring Damper System

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumtrapz

def inp_step(t):

	if t < 10:
		return 0
	elif (t > 10) and (t < 11):
		return 10
	else:
		return 0

def MSD(x0 = 0, dx0 = 0): #Mass spring damper system
	m = 10
	d = 20
	k = 2

	t_start = 0
	t_stop = 100
	t_step = 0.001
	t = list(np.arange(t_start,t_stop,t_step))

	A = np.array([[0, 1],[-k/m, -d/m]])
	B = np.array([[0], [1/m]])

	def sys(t,X):
		dX = np.dot(A,X) + np.dot(B,inp_step(t))
		return dX

	sol = solve_ivp(sys,[t[0], t[-1]],[x0,dx0], vectorized = True, t_eval = t, dense_output = True, max_step = t_step)
	acc = np.diff(sol.y[1,:])/np.diff(t)
	return  sol, acc



#solve
sol, acc = MSD()

#input
inp = []
for i in range(len(sol.t)):
	inp.append(inp_step(sol.t[i]))


plt.figure()
plt.subplot(312)
plt.plot(sol.t, sol.y[1,:])
plt.grid()
plt.ylabel('dx')
plt.subplot(313)
plt.plot(sol.t,sol.y[0,:])
plt.grid()
plt.ylabel('x')
plt.subplot(311)
plt.plot(sol.t[:-1],acc)
plt.grid()
plt.ylabel('ddot_x')
plt.xlabel('Time [s]')

plt.figure()
plt.plot(sol.t, inp)
plt.ylabel('Input [F]')
plt.xlabel('Time [s]')
plt.grid()
plt.show()


