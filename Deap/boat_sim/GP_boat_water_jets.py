"""
Testing out the new LS alg for the boat sim
Also trying to include OLS
"""

import operator
import math
import random

import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from sympy import sympify, cos, sin
from scipy import signal

import pygraphviz as pgv
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumtrapz
import sys
sys.path.insert(0, '/home/gislehalv/Master/scripts')
import my_lib
from scipy import optimize
from pytictoc import TicToc


#time
sim_time = [0, 300, 0.1]


#--inputs--#
def inp_step_x(t,states):
	if t < 5:
		return [[0],[0],[0]]
	else:
		return [[4000],[0],[0]]
def inp_step_series(t,states):
	if t < 5:
		return [[0],[0],[0]]

	elif t < 20:
		return [[4000],[0],[0]]

	elif t < 40:
		return [[0],[1000],[0]]

	elif t < 60:
		return [[0],[0],[1000]]

	elif t < 80:
		return [[0],[0],[-1000]]

	elif t < 100:
		return [[0],[-1000],[1000]]

	elif t < 120:
		return [[2000],[0],[0]]

	elif t < 140:
		return [[0],[0],[-200]]

	elif t < 160:
		return [[300],[0],[0]]

	else:
		return [[0],[0],[0]]
def steps_and_square(t,states):
	if t < 5:
		return [[0],[0],[0]]

	elif t < 20:
		return [[4000],[0],[0]]

	elif t < 40:
		return [[0],[1000],[0]]

	elif t < 60:
		return [[0],[0],[1000]]

	elif t < 80:
		return [[0],[0],[-1000]]

	elif t < 100:
		return [[0],[-1000],[1000]]

	elif t < 120:
		return [[2000],[0],[0]]

	elif t < 140:
		return [[0],[0],[-200]]

	elif t < 160:
		return [[300],[0],[0]]

	elif t < 250:
		return [[0],[1500 * signal.square(t/5)],[700 * signal.square(t/3)]]

	elif t > 270 and t < 300:
		return [[1500 * signal.square(t/3) + 1500],[0],[0]]

	elif t > 320 and t < 400:
		return [[2000 * sin(t/4) +2000],[0],[0]]
	
	elif t > 420 and t < 460:
		return [[2*t],[0],[0]]

	else:
		return [[0],[0],[0]]
def inp_step_x_y_z(t,states):
	if t < 5:
		return [[0],[0],[0]]
	elif t < 55:
		return [[-2000],[0],[0]]
	elif t < 105:
		return [[0],[-500],[0]]
	elif t < 155:
		return [[0],[0],[-200]]	
	elif t < 205:
		return [[1000],[0],[0]]	
	elif t < 255:
		return [[0],[700],[0]]	
	elif t < 305:
		return [[0],[0],[200]]	
	else:
		return [[0],[0],[0]]


#WJ input
prev_jet_input = []
prev_noz_input = []
prev_t = []

def input_WJ_1(t, states, return_force = True):
	def jet_model(nu, jet_rpm, delta_nozzle):
		#constants
		lever_CGtowj_port = [-3.82, -0.475]
		lever_CGtowj_stb = [-3.82, 0.475]
		rpm_slew_rate = [2000, -2000]
		nozzle_slew_rate = [1.3464, -1.3464]
		rpm_min_max = [0, 2000]

		if jet_rpm > rpm_min_max[1]:
			jet_rpm = rpm_min_max[1]
		elif jet_rpm < rpm_min_max[0]:
			jet_rpm = rpm_min_max[0]

		
		if 0:#prev_jet_input and prev_noz_input:
			#rate limiter rpm
			prev_jet_input.append(jet_rpm)
			jet_now = prev_input[-1]
			jet_prev = prev_input[-2]

			rate = (jet_now - jet_prev)/(t_now-t_prev)
			if rate > rpm_slew_rate[0]:
				new_rpm = (t_now-t_prev)*rpm_slew_rate[0] + jet_prev
			elif rate < rpm_slew_rate[1]:
				new_rpm = (t_now-t_prev)*rpm_slew_rate[1] + jet_prev

			#rate limiter nozzle
			prev_noz_input.append(delta_nozzle)
			jet_now = prev_noz_input[-1]
			jet_prev = prev_noz_input[-2]
			rate = (noz_now - noz_prev)/(t_now-t_prev)

			if rate > nozzle_slew_rate[0]:
				new_rpm = (t_now-t_prev)*nozzle_slew_rate[0] + jet_prev
			elif rate < nozzle_slew_rate[1]:
				new_rpm = (t_now-t_prev)*nozzle_slew_rate[1] + jet_prev





		#rpm2thrust
		speed = nu[0] * 1.94384 # knots 
		a0 = 6244.15
		a1 = -178.46
		a2 = 0.881043
		thrust_unscaled = a0 + a1*speed + a2*(speed**2)

		r0 = 85.8316
		r1 = -1.7935
		r2 = 0.00533
		rpm_scale = 1/4530*(r0 + r1*jet_rpm + r2 * (jet_rpm **2))

		thrust = rpm_scale * thrust_unscaled


		#waterjet port
		#force
		Fx = thrust*np.cos(delta_nozzle)
		Fy = thrust*np.sin(delta_nozzle)
		#moment
		Nz_port = (lever_CGtowj_port[0]*Fy)- (lever_CGtowj_port[1]*Fx)
		Nz_stb = (lever_CGtowj_stb[0]*Fy)- (lever_CGtowj_stb[1]*Fx)

		#tau_b_port = [Fx, Fy, Nz_port]
		#tau_b_stb = [Fx, Fy, Nz_stb]

		tau_b =  [2*Fx, 2*Fy, Nz_port + Nz_stb]#np.add(tau_b_port, tau_b_stb)
		#prev_jet_input.append(jet_rpm)
		#prev_noz_input.append(delta_nozzle)
		return tau_b

	nu = states[3:6]

	input_1 = True
	input_2 = False
	input_3 = False
	if input_1:    ### input 1 in the report
		if t <= 20:
			delta_nozzle = 0
			jet_rpm = 500 
		elif t <= 200:
			jet_rpm =  500 * signal.square(t/3) +1000
		else:
			jet_rpm =  1000 + 2*t

		if t > 50 and t < 100:
			delta_nozzle = 0.1*signal.square(t/3)
		elif t >= 100 and t < 150:
			delta_nozzle = 0.05*signal.square(t/2)
		elif t >= 150 and t < 200:
			delta_nozzle = 0.2*signal.square(t/4)
		elif t >= 200 and t < 250:
			delta_nozzle = 0.01*signal.square(t/2)	
		else:
			delta_nozzle = 0
	if input_2:
		delta_nozzle = 0
		jet_rpm = 0
		
	if input_3:   ### input 4 in the report
		delta_nozzle = 0.1 -t*0.001
		jet_rpm =  100 + 4*t

	prev_t.append(t)
	tau_b = jet_model(nu, jet_rpm, delta_nozzle)

	if return_force == True:
		return tau_b
	else:
		return [delta_nozzle,jet_rpm]

def input_WJ_2(t, states, return_force = True):
	def jet_model(nu, jet_rpm, delta_nozzle):
		#constants
		lever_CGtowj_port = [-3.82, -0.475]
		lever_CGtowj_stb = [-3.82, 0.475]
		rpm_slew_rate = [2000, -2000]
		nozzle_slew_rate = [1.3464, -1.3464]
		rpm_min_max = [0, 2000]

		if jet_rpm > rpm_min_max[1]:
			jet_rpm = rpm_min_max[1]
		elif jet_rpm < rpm_min_max[0]:
			jet_rpm = rpm_min_max[0]

		
		if 0:#prev_jet_input and prev_noz_input:
			#rate limiter rpm
			prev_jet_input.append(jet_rpm)
			jet_now = prev_input[-1]
			jet_prev = prev_input[-2]

			rate = (jet_now - jet_prev)/(t_now-t_prev)
			if rate > rpm_slew_rate[0]:
				new_rpm = (t_now-t_prev)*rpm_slew_rate[0] + jet_prev
			elif rate < rpm_slew_rate[1]:
				new_rpm = (t_now-t_prev)*rpm_slew_rate[1] + jet_prev

			#rate limiter nozzle
			prev_noz_input.append(delta_nozzle)
			jet_now = prev_noz_input[-1]
			jet_prev = prev_noz_input[-2]
			rate = (noz_now - noz_prev)/(t_now-t_prev)

			if rate > nozzle_slew_rate[0]:
				new_rpm = (t_now-t_prev)*nozzle_slew_rate[0] + jet_prev
			elif rate < nozzle_slew_rate[1]:
				new_rpm = (t_now-t_prev)*nozzle_slew_rate[1] + jet_prev





		#rpm2thrust
		speed = nu[0] * 1.94384 # knots 
		a0 = 6244.15
		a1 = -178.46
		a2 = 0.881043
		thrust_unscaled = a0 + a1*speed + a2*(speed**2)

		r0 = 85.8316
		r1 = -1.7935
		r2 = 0.00533
		rpm_scale = 1/4530*(r0 + r1*jet_rpm + r2 * (jet_rpm **2))

		thrust = rpm_scale * thrust_unscaled


		#waterjet port
		#force
		Fx = thrust*np.cos(delta_nozzle)
		Fy = thrust*np.sin(delta_nozzle)
		#moment
		Nz_port = (lever_CGtowj_port[0]*Fy)- (lever_CGtowj_port[1]*Fx)
		Nz_stb = (lever_CGtowj_stb[0]*Fy)- (lever_CGtowj_stb[1]*Fx)

		#tau_b_port = [Fx, Fy, Nz_port]
		#tau_b_stb = [Fx, Fy, Nz_stb]

		tau_b =  [2*Fx, 2*Fy, Nz_port + Nz_stb]#np.add(tau_b_port, tau_b_stb)
		#prev_jet_input.append(jet_rpm)
		#prev_noz_input.append(delta_nozzle)
		return tau_b

	nu = states[3:6]

	input_1 = False
	input_2 = False
	input_3 = False
	input_4 = True
	if input_1:
		if t <= 20:
			delta_nozzle = 0
			jet_rpm = 500 
		else:
			jet_rpm =  500 * signal.square(t/3) +1000

		if t > 50 and t < 55:
			delta_nozzle = 0.1
		elif t >= 55 and t < 65:
			delta_nozzle = -0.1
		elif t >= 90 and t < 92:
			delta_nozzle = 0.1
		elif t >= 95 and t < 105:
			delta_nozzle = -0.1
		#elif t >= 75 and t < 80:
		#	delta_nozzle = 0.1
		else:
			delta_nozzle = 0
	
	if input_2:
		delta_nozzle = 0
		jet_rpm = 0
		
	if input_3:
		delta_nozzle = 0
		jet_rpm =  500 * signal.square(t/3) +1000

	if input_4:   #### input 2 in the report
		if t <= 20:
			delta_nozzle = 0
			jet_rpm = 500 
		else:
			jet_rpm =  700 * signal.square(t/3) +1000

		if t > 50 and t < 55:
			delta_nozzle = 0.05
		elif t >= 55 and t < 65:
			delta_nozzle = -0.05
		elif t >= 90 and t < 92:
			delta_nozzle = 0.05
		elif t >= 95 and t < 105:
			delta_nozzle = -0.05
		elif t >= 180 and t < 185:
			delta_nozzle = 0.1
		elif t >= 190 and t < 200:
			delta_nozzle = -0.1
		else:
			delta_nozzle = 0
	
	prev_t.append(t)
	tau_b = jet_model(nu, jet_rpm, delta_nozzle)

	if return_force == True:
		return tau_b
	else:
		return [delta_nozzle,jet_rpm]
 
def input_WJ_3(t, states, return_force = True):
	def jet_model(nu, jet_rpm, delta_nozzle):
		#constants
		lever_CGtowj_port = [-3.82, -0.475]
		lever_CGtowj_stb = [-3.82, 0.475]
		rpm_slew_rate = [2000, -2000]
		nozzle_slew_rate = [1.3464, -1.3464]
		rpm_min_max = [0, 2000]

		if jet_rpm > rpm_min_max[1]:
			jet_rpm = rpm_min_max[1]
		elif jet_rpm < rpm_min_max[0]:
			jet_rpm = rpm_min_max[0]

		
		if 0:#prev_jet_input and prev_noz_input:
			#rate limiter rpm
			prev_jet_input.append(jet_rpm)
			jet_now = prev_input[-1]
			jet_prev = prev_input[-2]

			rate = (jet_now - jet_prev)/(t_now-t_prev)
			if rate > rpm_slew_rate[0]:
				new_rpm = (t_now-t_prev)*rpm_slew_rate[0] + jet_prev
			elif rate < rpm_slew_rate[1]:
				new_rpm = (t_now-t_prev)*rpm_slew_rate[1] + jet_prev

			#rate limiter nozzle
			prev_noz_input.append(delta_nozzle)
			jet_now = prev_noz_input[-1]
			jet_prev = prev_noz_input[-2]
			rate = (noz_now - noz_prev)/(t_now-t_prev)

			if rate > nozzle_slew_rate[0]:
				new_rpm = (t_now-t_prev)*nozzle_slew_rate[0] + jet_prev
			elif rate < nozzle_slew_rate[1]:
				new_rpm = (t_now-t_prev)*nozzle_slew_rate[1] + jet_prev





		#rpm2thrust
		speed = nu[0] * 1.94384 # knots 
		a0 = 6244.15
		a1 = -178.46
		a2 = 0.881043
		thrust_unscaled = a0 + a1*speed + a2*(speed**2)

		r0 = 85.8316
		r1 = -1.7935
		r2 = 0.00533
		rpm_scale = 1/4530*(r0 + r1*jet_rpm + r2 * (jet_rpm **2))

		thrust = rpm_scale * thrust_unscaled


		#waterjet port
		#force
		Fx = thrust*np.cos(delta_nozzle)
		Fy = thrust*np.sin(delta_nozzle)
		#moment
		Nz_port = (lever_CGtowj_port[0]*Fy)- (lever_CGtowj_port[1]*Fx)
		Nz_stb = (lever_CGtowj_stb[0]*Fy)- (lever_CGtowj_stb[1]*Fx)

		#tau_b_port = [Fx, Fy, Nz_port]
		#tau_b_stb = [Fx, Fy, Nz_stb]

		tau_b =  [2*Fx, 2*Fy, Nz_port + Nz_stb]#np.add(tau_b_port, tau_b_stb)
		#prev_jet_input.append(jet_rpm)
		#prev_noz_input.append(delta_nozzle)
		return tau_b

	nu = states[3:6]

	input_1 = False
	input_2 = True
	input_3 = False
	input_4 = False
	if input_1:
		if t <= 20:
			delta_nozzle = 0
			jet_rpm = 500 
		else:
			jet_rpm =  500 * signal.square(t/3) +1000

		if t > 50 and t < 55:
			delta_nozzle = 0.1
		elif t >= 55 and t < 65:
			delta_nozzle = -0.1
		elif t >= 90 and t < 92:
			delta_nozzle = 0.1
		elif t >= 95 and t < 105:
			delta_nozzle = -0.1
		#elif t >= 75 and t < 80:
		#	delta_nozzle = 0.1
		else:
			delta_nozzle = 0
	
	if input_2:  #### input 3 in the report
		delta_nozzle = 0.1 * np.sin(t)
		jet_rpm = 800*np.sin(t/3) + 800
		
	if input_3:
		delta_nozzle = 0
		jet_rpm =  500 * signal.square(t/3) +1000

	if input_4:
		if t <= 20:
			delta_nozzle = 0
			jet_rpm = 500 
		else:
			jet_rpm =  700 * signal.square(t/3) +1000

		if t > 50 and t < 55:
			delta_nozzle = 0.05
		elif t >= 55 and t < 65:
			delta_nozzle = -0.05
		elif t >= 90 and t < 92:
			delta_nozzle = 0.05
		elif t >= 95 and t < 105:
			delta_nozzle = -0.05
		elif t >= 180 and t < 185:
			delta_nozzle = 0.1
		elif t >= 190 and t < 200:
			delta_nozzle = -0.1
		else:
			delta_nozzle = 0
	
	prev_t.append(t)
	tau_b = jet_model(nu, jet_rpm, delta_nozzle)

	if return_force == True:
		return tau_b
	else:
		return [delta_nozzle,jet_rpm]
 

### RUN SIM  ###
def boat_simulation_wj(input_func, time = sim_time, init_cond = [0, 0, 0, 0, 0, 0]):

	## System ###
	def sys(t,X):

		
		tauB = np.zeros((3,1))
		tauB = input_func(t,X)

		# eta = np.zeros((3,1))
		# eta[0,0] = X[0]
		# eta[1,0] = X[1]
		# eta[2,0] = X[2]

		nu = np.zeros((3,1))
		nu[0,0] = X[3]
		nu[1,0] = X[4]
		nu[2,0] = X[5]

		#x = eta[0];
		#y = eta[1];
		yaw = X[2];


		u = nu[0];
		v = nu[1];
		r = nu[2];


		# mass and moment of inertia
		m = 4935.14
		Iz = 20928 #TODO Pz, sjekk at dette stemmer. Burde være Px?????

		# center of gravity
		xg = 0
		yg = 0

		# added mass
		Xdu = 0
		Ydv = 0
		Ydr = 0
		Ndv = 0
		Ndr = 0

		# damping (borrowed from Loe 2008)
		Xu = -50
		Yv = -200
		Yr = 0
		Nr = -1281

		Xuu = -135*1.8 #TODO gang med 1.8?? Hvorfor gjør Thomas det
		Yvv = -2000
		T = 4
		K = 0.5
		Nrr = -Iz*1.1374*K/T # TODO eller 0??

		# transformation matrix, equal to rotation matrix about z-axis
		J = np.array([[np.cos(yaw), -np.sin(yaw), 0],
		     [np.sin(yaw), np.cos(yaw),  0],
		     [0,        0,         1,]])

		# rigid body mass
		M_RB = np.array([[m, 0, 0],
		        [0, m, 0],
		        [0, 0, Iz]])

		# hydrodynamic added body mass
		M_A = -np.array([[Xdu, 0, 0],
		        [0, Ydv, Ydr],
		        [0, Ndv, Ndr]])
		 
		# total mass
		M = M_RB + M_A

		#Coriolis
		C_RB_g = np.zeros((3,3))
		C_RB_g[0,2] = -m*(xg*r+v)
		C_RB_g[2,0] = -C_RB_g[0,2]
		C_RB_g[1,2] = m*u
		C_RB_g[2,1] = -C_RB_g[1,2]

		C_A_g = np.zeros((3,3))
		C_A_g[0,2] = (Ydv*v)+(Ydr*r)
		C_A_g[2,0] = -C_A_g[0,2]
		C_A_g[1,2] = Xdu*u
		C_A_g[2,1] = -C_A_g[1,2]

		C_g = np.add(C_RB_g, C_A_g)

		#Linear damping
		Dl_g = -np.array([[Xu, 0, 0],
						[0, Yv, 0],
						[0, 0, Nr]])
		Dl_g[1,2] = -Yr;
		Dl_g[2,1] = -Nr;

		#Nonlinear damping
		Dnl_g = - np.array([[Xuu*np.abs(u), 0, 0],
						[0, Yvv*abs(v), 0],
						[0, 0, Nrr*abs(r)]])
		

		D_g = np.add(Dl_g, Dnl_g)

		

		eta_dot = np.dot(J,nu);
		nu_dot = np.dot(np.linalg.inv(M), (tauB - np.dot(C_g, nu) - np.dot(D_g, nu)))


		out = np.concatenate((eta_dot, nu_dot))
		

		return out

	#time
	t_start, t_stop, t_step = time 
	t = list(np.arange(t_start, t_stop, t_step))

	#solve
	sol = solve_ivp(sys, [t[0], t[-1]], init_cond, vectorized = True, max_step = t_step, t_eval = t)

	#input to the sys
	inp = np.zeros((3,len(sol.t)))
	for i,time in enumerate(sol.t):
		inp[:,i] = np.squeeze(input_func(time,sol.y[:,i]))
	
	plt.figure()
	plt.subplot(311)
	plt.plot(sol.t, inp[0,:])
	plt.ylabel('tau x [N]')
	plt.subplot(312)
	plt.plot(sol.t, inp[1,:])
	plt.ylabel('tau y [N]')
	plt.subplot(313)
	plt.plot(sol.t, inp[2,:])
	plt.ylabel('tau z [Nm]')

	#get the derivatives
	sol_dot = np.zeros((np.shape(sol.y)[0], np.shape(sol.y)[1]))
	for tmp, i in enumerate(sol.t):

		#simulate the derivatives
		sol_dot[:,tmp] = np.squeeze(sys(i, np.expand_dims(sol.y[:,tmp], axis = 1)))


		#explicitly solve the equations:
		# sol_dot[0,tmp] = 2.026e-4*inp[0,tmp] + sol.y[4,tmp]*sol.y[5,tmp] - 0.0101*sol.y[3,tmp] -0.0492*np.abs(sol.y[3,tmp])*sol.y[3,tmp]
		# sol_dot[1,tmp] = 2.026e-4*inp[1,tmp] - sol.y[3,tmp]*sol.y[5,tmp] - 0.045*sol.y[4,tmp] - 0.4052*np.abs(sol.y[4,tmp])*sol.y[4,tmp]
		# sol_dot[2,tmp]= 4.78e-5*inp[2,tmp] -0.0612*(sol.y[4,tmp]+sol.y[5,tmp]) - 0.1422*np.abs(sol.y[5,tmp])*sol.y[5,tmp]

		# old: sol_dot[5,tmp]= 4.78e-5*inp[2,tmp] -0.2595*(sol.y[4,tmp]+sol.y[5,tmp]) - 0.6027*np.abs(sol.y[5,tmp])*sol.y[5,tmp]


	#wj input 
	inp = np.zeros((2,len(sol.t)))
	for i, time in enumerate(sol.t):
		inp[:,i] = np.squeeze(input_func(time, sol.y[:,i], return_force = False))

	output = np.concatenate((sol.y[3:],sol_dot[3:],inp,sol.t.reshape(1,-1)), axis = 0)


	#x-y plot
	plt.figure()
	plt.title('X-Y plot')
	plt.plot(sol.y[0], sol.y[1])


	return output

X = boat_simulation_wj(input_WJ_1, time = sim_time)

X_val = boat_simulation_wj(input_WJ_2, time = sim_time)

X_test = boat_simulation_wj(input_WJ_3, time = sim_time)

#my_lib.boat_sim_plot_wj(X, show = True)


#exit()


###  what eq to find.
solve_for_du = False 
solve_for_dv = False
solve_for_dr = True
if solve_for_du:
	y = X[3]
	y_val = X_val[3]
	y_test = X_test[3]
if solve_for_dv:
	y = X[4]
	y_val = X_val[4]
	y_test = X_test[4]
if solve_for_dr:
	y = X[5]
	y_val = X_val[5]
	y_test = X_test[5]

#new functoins 
def square(a):
	return a**2


#Operators
pset = gp.PrimitiveSet("MAIN", 5)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.abs, 1)
pset.addPrimitive(np.sin, 1)
pset.addPrimitive(np.cos, 1)
#pset.addPrimitive(square, 1)

#Variable names 
pset.renameArguments(ARG0='u')
pset.renameArguments(ARG1='v')
pset.renameArguments(ARG2='r')
pset.renameArguments(ARG3='delta_t')
pset.renameArguments(ARG4='delta_n')


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

##Toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


#works for arity 0, 1 and 2 and only for add (not sub)
def split_tree(individual):
	
	def tree_trav(individual):
		nodes, edges, labels = gp.graph(individual)
		main_roots = []

		#is the first root add or sub
		if labels[0] == 'add':# or labels[0] == 'sub':
			main_roots.append(nodes[0])
		else:
			return None

		#find the main roots
		for node in sorted(nodes):
			if labels[node] == 'add':# or labels[node] == 'sub':
				if node not in main_roots:
					for edge in edges: 
						if node == edge[1] and edge[0] in main_roots: #if the previus node is in roots
							main_roots.append(node)

		for root in main_roots:
			for edge in edges:
				if edge[0] in main_roots:
					if edge[1] not in main_roots and edge[1] not in roots:					
						roots.append(edge[1])
		return main_roots

	def ext_funcs(individual):
		for root in roots:

			#divide into substrings
			F = individual[individual.searchSubtree(root)]

			string = ''


			if len(F) == 1:
				string = F[0].format()

			else:		
	
				for item in F:
					if item.arity == 0:
						string = string.split(' ')[0] + item.format() + ' '.join(string.split(' ')[1:])
					if item.arity == 1:
						if len(string) > 0:
							string = string.split(' ')[0] + item.name + '( )' + ' '.join(string.split(' ')[1:])
						else:
							string = item.name + '( )'

					if item.arity == 2:
						if len(string) > 0:
							string = string.split(' ')[0] + item.name + '( , )' + ' '.join(string.split(' ')[1:])
						else:
							string = item.name + '( , )' 


			str_list.append(string)
			new_ind = gp.PrimitiveTree.from_string(string,pset)
			func1 = toolbox.compile(expr=new_ind)
			subtree_list.append(func1)


	subtree_list = []
	str_list = []
	roots = []
	main_roots = tree_trav(individual)
	if main_roots == None:
		str_list.append(str(individual))
		return [toolbox.compile(expr=individual)], str_list

	ext_funcs(individual)
	return subtree_list, str_list


#either return_str = True or plot_result = True, not both. 

def eval_fit_new_w_constant(individual, u, v, r, delta_t, delta_n, y, return_str = False, plot_result = False):
	#print('individual: ',individual)
	funcs, str_list = split_tree(individual)
	F_list = []
	

	#top root is not 'add'
	if len(funcs) == 1:

		F = funcs[0](u, v, r, delta_t, delta_n)
		F_trans = np.transpose(F)

		p = np.dot(np.dot(F_trans,F),np.dot(F_trans,y)) 
		p = [p]


	#top root is 'add'
	else:
		for func in funcs:
			F_list.append(func)
		F = np.zeros((len(y), len(F_list)))

		for i, function in enumerate(F_list):
			F[:,i] = np.squeeze(function(u, v, r, delta_t, delta_n))

		F_trans = np.transpose(F)
		try:
			p = np.dot(np.linalg.inv(np.dot(F_trans,F)),np.dot(F_trans,y))  
		except:
			#print('Singular Matrix for: ', individual)
			mse = 1000 # large number
			return(mse,)

	tot_func = np.zeros((len(y)))

	for i, func in enumerate(funcs):
		tot_func = np.add(tot_func, p[i]*func(u, v, r, delta_t, delta_n))


	mse = math.fsum((y-tot_func)**2)/len(y)


	#return the simplified eq
	if return_str:
		locals = {
			'mul': lambda x, y : x * y,
			'add': lambda x, y : x + y,
			'add3': lambda x, y, z: x+y+z,
			'sub': lambda x, y : x - y,
			'protectedDiv': lambda x, y: x / y,
			'neg': lambda x: -x,
			'sin': lambda x: sin(x),
			'cos': lambda x: cos(x),
			'abs': lambda x: np.abs(x)#x if x >= 0 else -x
		}
		tot_str = ''
		for i, func_str in enumerate(str_list):
			tot_str = tot_str +'+'+ str(p[i])+ '*' +func_str
		function_string = sympify(tot_str,locals = locals)
		return function_string

	if plot_result:
		plt.figure()
		plt.plot(tot_func)
		plt.plot(y)
		plt.xlabel('Samples')
		plt.legend(['Predicted', 'Ground Truth'])
		plt.grid()


	return(mse,)



#test function
if 0:
	individual = 'add(add(u,v),mul(r,tau_z))'
	individual = gp.PrimitiveTree.from_string(individual,pset)
	eval_fit_new(individual, u = X[0], v = X[1], r = X[2], tau_x = X[6], tau_y = X[7], tau_z = X[8], du = X[3], return_str = False)
	print(eval_fit_new(individual, u = X[0], v = X[1], r = X[2], tau_x = X[6], tau_y = X[7], tau_z = X[8], du = X[3], return_str = True))
	exit()




toolbox.register("evaluate", eval_fit_new_w_constant, u = X[0], v = X[1], r = X[2], delta_t = X[-2,:], delta_n = X[-3,:], y = y, return_str = False)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))

### main algorithm ##
#constants

pop_size = 5000
mate_prob = 0.5
mut_prob = 0.3
generations = 100

#parsimony coefficient
#if MSE_pars:
#	pc = 0.2

pop = toolbox.population(n=pop_size)
hof = tools.HallOfFame(1)
logbook = tools.Logbook()

lambda_ = int(pop_size/2)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("max", np.max)
stats.register("min", np.min)


val_acc = []
train_acc = []

for gen in range(0,generations):
	pop = algorithms.varOr(pop, toolbox, lambda_, mate_prob, mut_prob)
	invalid_ind = [ind for ind in pop if not ind.fitness.valid]

	#print(len(invalid_ind))
	fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)	
	for ind, fit in zip(invalid_ind, fitnesses):
		ind.fitness.values = fit
	hof.update(pop)

	record = stats.compile(pop)
	logbook.record(gen=gen, evals=len(invalid_ind), **record)
	pop = toolbox.select(pop, k=len(pop))
	print('Generation:',gen)
	print('Best test set score: ',record['min'])

	train_acc.append(record['min'])

	val_score = eval_fit_new_w_constant(hof[0], u = X_val[0], v = X_val[1], r = X_val[2], delta_t = X_val[-2,:], delta_n = X_val[-3,:], y = y_val, return_str = False)[0]
	val_acc.append(val_score)
	print('validation score: ',val_score)
	

	#save best val



	#test result on validation set
	if record['min'] < 1e-8:
		mse = eval_fit_new_w_constant(hof[0], u = X_val[0], v = X_val[1], r = X_val[2], delta_t = X_val[-2,:], delta_n = X_val[-3,:], y = y_val, return_str = False)
		print('mse for validation: ', mse)
		if mse[0] < 1e-8:
			#print clean eq, and lisp eq
			print('Final result:',eval_fit_new_w_constant(hof[0], u = X_val[0], v = X_val[1], r = X_val[2], delta_t = X_val[-2,:], delta_n = X_val[-3,:], y = y_val, return_str = True))
			print(hof[0])

			#plot
			eval_fit_new_w_constant(hof[0], u = X_val[0], v = X_val[1], r = X_val[2], delta_t = X_val[-2,:], delta_n = X_val[-3,:], y = y_val, plot_result = True)
			plt.title('Validation set')

			eval_fit_new_w_constant(hof[0], u = X_test[0], v = X_test[1], r = X_test[2], delta_t = X_test[-2,:], delta_n = X_test[-3,:], y = y_test, plot_result = True)
			plt.title('Test set')

			#history plot
			plt.figure()
			plt.semilogy(list(range(len(val_acc))),val_acc)
			plt.semilogy(list(range(len(val_acc))),train_acc)
			plt.grid()
			plt.legend(['Validation accuracy', 'Training accuracy'])
			plt.show()
			exit()



print('Reached the max number of generations')
print('Best equation:',eval_fit_new_w_constant(hof[0], u = X_val[0], v = X_val[1], r = X_val[2], delta_t = X_val[-2,:], delta_n = X_val[-3,:], y = y_val, return_str = True))
eval_fit_new_w_constant(hof[0], u = X_val[0], v = X_val[1], r = X_val[2], delta_t = X_val[-2,:], delta_n = X_val[-3,:], y = y_val, plot_result = True)
plt.show()

##history plot
# plt.figure()
# plt.semilogy(logbook.select('gen'),logbook.select('min'))
# plt.xlabel('Generations')
# plt.ylabel('Mean Sqaure Error')
# plt.grid()
# plt.show()









