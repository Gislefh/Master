## script to test the new jater jet model and with data from matlab###

import numpy as np

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumtrapz
from scipy import signal
##  -- import data -- 
# file_path = '/home/gislehalv/Master/Data/data_fom_MatlabSim/du_data_wj3.csv'
# data = np.loadtxt(file_path, delimiter = ',')
# ## data = [u,v,r,U,du,dv,dr,dU,force_x,force_y,force_z,jet_rpm,nozzle_angle,time]


# u = data[:, 0]
# v = data[:, 1]
# r = data[:, 2]

# fx = data[:, 8]
# fy = data[:, 9]
# fz = data[:, 10]

# jet_rpm = data[:, 11]
# nozzle_angle = data[:, 12]

# du = data[:, 4]
# dv = data[:, 5]
# dr = data[:, 6]

# sim_t = data[:, -1]
sim_time = [0, 300, 0.1]


test_t = []
e = []
def inp_data2inp(t,states, return_force = True):
	def jet_model(nu, jet_rpm, delta_nozzle):
		#constants
		lever_CGtowj_port = [-3.82, -0.475]
		lever_CGtowj_stb = [-3.82, 0.475]
		#rpm_slew_rate = [2000, -2000]
		#nozzle_slew_rate = [1.3464, -1.3464]
		#rpm_min_max = [0, 2000]

		#if jet_rpm > rpm_min_max[1]:
		#	jet_rpm = rpm_min_max[1]
		#elif jet_rpm < rpm_min_max[0]:
		#	jet_rpm = rpm_min_max[0]

		
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

		tau_b =  [Fx+Fx, Fy+Fy, Nz_port + Nz_stb]#np.add(tau_b_port, tau_b_stb)
		#prev_jet_input.append(jet_rpm)
		#prev_noz_input.append(delta_nozzle)
		return tau_b

	nu = states[3:6]

	#find closest match wrt time
	idx = (np.abs(sim_t - t)).argmin()
	na = nozzle_angle[idx]
	jr = jet_rpm[idx]
	tau_b = jet_model(nu, jr, na)

	test_t.append(t)

	fx_test =  2*np.multiply((118.31 - 2.47*jet_rpm[idx] + 7.35e-3 * np.multiply(jet_rpm[idx],jet_rpm[idx]) - 6.56*u[idx] + 0.14 *np.multiply(jet_rpm[idx],u[idx]) - 4.07e-4* np.multiply(np.multiply(jet_rpm[idx],jet_rpm[idx]),u[idx]) + 0.063 *np.multiply(u[idx],u[idx]) - 1.31e-3 *np.multiply(np.multiply(u[idx],u[idx]),jet_rpm[idx]) + 3.90e-6*np.multiply(np.multiply(u[idx],u[idx]),np.multiply(jet_rpm[idx],jet_rpm[idx])) ),np.cos(nozzle_angle[idx]))

	e.append(fx_test- fx[idx]) 

	if return_force == True:
		return tau_b
	else:
		return [na,jr]


def boat_simultiplyation_wj(input_func, time = sim_time, init_cond = [0, 0, 0, 0, 0, 0]):

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
						[Nrr*abs(r), 0, 0]])
		

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
	

	#get the derivatives
	sol_dot = np.zeros((np.shape(sol.y)[0], np.shape(sol.y)[1]))
	for tmp, i in enumerate(sol.t):

		#simultiplyate the derivatives
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

def input_test(t,states, return_force = True):
	def jet_model(nu, jet_rpm, delta_nozzle):
		#constants
		lever_CGtowj_port = [-3.82, -0.475]
		lever_CGtowj_stb = [-3.82, 0.475]
		#rpm_slew_rate = [2000, -2000]
		#nozzle_slew_rate = [1.3464, -1.3464]
		#rpm_min_max = [0, 2000]

		#if jet_rpm > rpm_min_max[1]:
		#	jet_rpm = rpm_min_max[1]
		#elif jet_rpm < rpm_min_max[0]:
		#	jet_rpm = rpm_min_max[0]

		
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

		tau_b =  [Fx+Fx, Fy+Fy, Nz_port + Nz_stb]#np.add(tau_b_port, tau_b_stb)
		#prev_jet_input.append(jet_rpm)
		#prev_noz_input.append(delta_nozzle)
		return tau_b

	nu = states[3:6]
	jr = 1000
	na = signal.square(t/3)*0.1
	tau_b = jet_model(nu, jr, na)

	if return_force == True:
		return tau_b
	else:
		return [na,jr]


X = boat_simultiplyation_wj(input_test, time = sim_time)

jet_rpm = X[-2]
nozzle_angle = X[-3]
#output = np.concatenate((sol.y[3:],sol_dot[3:],inp,sol.t.reshape(1,-1)), axis = 0)
du_eq = X[1]*X[2]- 0.0101*X[0] -0.0492*np.abs(X[0])*X[0] + 2.026e-4* 2*np.multiply((118.31 - 2.47*jet_rpm + 7.35e-3 * np.multiply(jet_rpm,jet_rpm) - 6.56*X[0] + 0.14 *np.multiply(jet_rpm,X[0]) - 4.07e-4* np.multiply(np.multiply(jet_rpm,jet_rpm),X[0]) + 0.063 *np.multiply(X[0],X[0]) - 1.31e-3 *np.multiply(np.multiply(X[0],X[0]),jet_rpm) + 3.90e-6*np.multiply(np.multiply(X[0],X[0]),np.multiply(jet_rpm,jet_rpm)) ),np.cos(nozzle_angle))



plt.figure()
plt.plot(X[-1], du_eq)
plt.plot(X[-1],X[3])
plt.legend(['eq', 'sim'])


plt.show()


# plt.figure()
# plt.subplot(311)
# plt.plot(sim_t,du)
# plt.plot(X[-1],X[3])
# plt.legend(['du', 'du python sim'])
# plt.subplot(312)
# plt.plot(sim_t,dv)
# plt.plot(X[-1],X[4])
# plt.legend(['dv', 'dv python sim'])
# plt.subplot(313)
# plt.plot(sim_t,dr)
# plt.plot(X[-1],X[5])
# plt.legend(['dr', 'dr python sim'])


# # plt.figure()
# # plt.plot(e)
# plt.show()


######## equations 
#jet_rpm = data[:, 11]
#nozzle_angle = data[:, 12]



#\Dot{v} &=  u r - 0.045v - 0.4052|v|v + (2.52 - 0.14u + 0.00134u^2 - 1.60\cdot10^{-7}\delta_t + 4.74\cdot10^{-10}\delta_t^2)\sin(\delta_n) \label{eq:boatsim_wj_complete2}\\

#\Dot{r} &= -0.2595(v+r) - 0.6027|r|r + (-4.61\cdot10^{-4} + 2.56\cdot10^{-5}u -2.45\cdot10^{-7}u^2 + 2.93\cdot10^{-11}\delta_t -8.67\cdot10^{-14}\delta_t^2)\sin(\delta_n)