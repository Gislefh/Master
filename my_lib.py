#My Lib
import numpy as np
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor

from scipy.integrate import solve_ivp, cumtrapz
import pydotplus
from PIL import Image
import io
from scipy import signal
from scipy import optimize

from sympy import sympify, cos, sin

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


import rosbag
import time
from scipy import interpolate



"""-------- MASS SPRING DAMPER ---------
_IN_
time: 	[t_start, t_stop, t_step] -> start time, stop time, step length
mdk:	[m, d, k] -> mass, damping, spring stiffness
tau:	input to the system -> either: 'sin', 'step' or square


_OUT_
t,ddx,dx,x,inp: sim time, d²x/d²t, dx/dt, x, input to the system

"""
def MSD(time = [0,10,0.1], mdk = [1,1,1], x0 = 0, dx0 = 0, tau = 'step', time_delay = 0, true_eq = False): #Mass spring damper system
	

	#----Inputs
	def inp_step(t):
		if t <= time_delay:
			t = 0
		else:
			t = t - time_delay
		if t <= 5:
			return 0
		#elif (t > 5) and (t < 10):
		#	return 10
		else:
			return 10

	def inp_sin(t):
		if t < time_delay:
			t = 0
		else:
			t = t - time_delay

		return np.sin(t)

	def inp_square(t):
		if t <= time_delay:
			t = 0
			return 0
		else:
			t = t - time_delay
		return signal.square(t/3)

	inp = []


	#----SYS
	#const
	m,d,k = mdk
	if true_eq:
		print('true eq: ',1/m,'*tau -',d/m,'*dx -',k/m,'*x')

	#time

	t_start, t_stop, t_step = time
	t = list(np.arange(t_start,t_stop,t_step))

	#matrices
	A = np.array([[0, 1],[-k/m, -d/m]])
	B = np.array([[0], [1/m]])


	if tau == 'step':
		def sys(t,X):
			dX = np.dot(A,X) + np.dot(B,inp_step(t))
			return dX

	elif tau == 'sin':
		def sys(t,X):
			dX = np.dot(A,X) + np.dot(B,inp_sin(t))
			return dX

	elif tau == 'square':
		def sys(t,X):
			dX = np.dot(A,X) + np.dot(B,inp_square(t))
			return dX

	else:
		print('not a valid input for < tau >')
		exit()

	sol = solve_ivp(sys,[t[0], t[-1]],[x0,dx0], vectorized = True,  max_step = t_step, t_eval = t)

	#delayed input

	for i in range(len(sol.t)):
		if tau == 'step': 
			inp.append(inp_step(sol.t[i]))
		elif tau == 'sin': 
			inp.append(inp_sin(sol.t[i]))
		elif tau == 'square': 
			inp.append(inp_square(sol.t[i]))



	#ddx
	acc = np.multiply((1/m),inp) - np.multiply((d/m),sol.y[1, :]) - np.multiply((k/m),sol.y[0, :])
	

	#input
	inp = []
	time_delay = 0
	for i in range(len(sol.t)):
		if tau == 'step': 
			inp.append(inp_step(sol.t[i]))
		elif tau == 'sin': 
			inp.append(inp_sin(sol.t[i]))
		elif tau == 'square': 
			inp.append(inp_square(sol.t[i]))

	inp = np.array(inp).reshape(-1,1)

	return  np.array(sol.t), np.array(acc).reshape(-1,1), np.array(sol.y[1, :]).reshape(-1,1), np.array(sol.y[0, :]).reshape(-1,1), inp


"""
 ------------------ FIND THE CONSTANTS FROM LSR AND CREATE NEW STRING - for MSD sys ----------------
_IN_ 
func: 				the function with arguments (dx, x, tau)
hof:				string to append the weights to
ddx, dx, x, tau: 	the variables

_OUT_ 
new_string: the new string
sol: 		weights from the inputs
"""
def new_string_from_LSR(func, hof, ddx, dx, x, tau):

	#find good constants
	def fun(X):
		se = (ddx - func(X[0]*dx, X[1]*x, X[2]*tau))**2
		se_1d = np.squeeze(se)
		return se_1d
	x0 = np.array([1,1,1])
	sol = optimize.least_squares(fun,x0)

	#Add the constants to the equation (new_str)
	tmp = str(hof)
	new_str = ''
	skip = 0
	for i in range(len(tmp)):
		if tmp[i] == 'd' and tmp[i+1] == 'x':
			new_str = new_str +"mul(dx,{:.7f})".format(sol.x[0])
			skip = 2

		elif tmp[i] == 'x' and tmp[i-1] != 'd':
			new_str = new_str +"mul(x,{:.7f})".format(sol.x[1])
			skip = 1

		elif tmp[i] == 't' and tmp[i+1] == 'a' and tmp[i+2] == 'u':
			new_str = new_str +"mul(tau,{:.7f})".format(sol.x[2])
			skip = 3

		elif skip == 0:
			new_str = new_str + tmp[i]

		if skip != 0:
			skip = skip - 1


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
	new_str = sympify(new_str,locals = locals)

	return new_str, sol 





"""-------- BOAT SIMULATION  ---------
_IN_
input: function that takes in time and states and returns a [3,1] vector for fx,fy,fz
time: 	[t_start, t_stop, t_step] -> start time, stop time, step length
init_cond: [x0, y0, yaw0, u0, v0, r0]

_OUT_
output: [u,v,r,du,dv,dr,fx,fy,fz,t], a [10,n] matrix

_NOTES_
Actual equation in component form:
du = 2.026e-4*fx + v*r - 0.0101*u -0.0492*np.abs(u)*u
dv = 2.026e-4*fy + u*r - 0.045*v - 0.4052*np.abs(v)*v
dr = 4.78e-5*fz -0.2595*(v+r) - 0.6027*np.abs(r)*r
"""
def boat_simulation(input, time = [0, 30, 0.01], init_cond = [0, 0, 0, 0, 0, 0]):

	## System ###
	def sys(t,X):

		
		tauB = np.zeros((3,1))
		tauB = input(t,X)

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
		C_A_g[0,2] = Ydv*v+Ydr*r
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
		# Dnl_g = - np.array([[Xuu*np.abs(u), 0, 0],
		# 				[0, Yvv*abs(v), 0],
		# 				[Nrr*abs(r), 0, 0]])
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
		inp[:,i] = np.squeeze(input(time,sol.y[:,i]))

	#get the derivatives
	sol_dot = np.zeros((np.shape(sol.y)[0], np.shape(sol.y)[1]))
	for tmp, i in enumerate(sol.t):
		#sol_dot[:,tmp] = np.squeeze(sys(i, sol.y[:,tmp]))
		sol_dot[:,tmp] = np.squeeze(sys(i, np.expand_dims(sol.y[:,tmp], axis = 1)))
		#sol_dot[3,tmp] = 2.026e-4*inp[0,tmp] + sol.y[4,tmp]*sol.y[5,tmp] - 0.0101*sol.y[3,tmp] -0.0492*np.abs(sol.y[3,tmp])*sol.y[3,tmp]
		#sol_dot[4,tmp] = 2.026e-4*inp[1,tmp] + sol.y[3,tmp]*sol.y[5,tmp] - 0.045*sol.y[4,tmp] - 0.4052*np.abs(sol.y[4,tmp])*sol.y[4,tmp]
		#sol_dot[5,tmp]= 4.78e-5*inp[2,tmp] -0.2595*(sol.y[4,tmp]+sol.y[5,tmp]) - 0.6027*np.abs(sol.y[5,tmp])*sol.y[5,tmp]


	#final output [u,v,r,du,dv,dr,fx,fy,fz,t]
	#output = np.concatenate((sol_dot,inp,sol.t.reshape(1,-1)), axis = 0)
	output = np.concatenate((sol.y[3:],sol_dot[3:],inp,sol.t.reshape(1,-1)), axis = 0)


	#x-y plot
	plt.figure()
	plt.title('X-Y plot')
	plt.plot(sol.y[0], sol.y[1])


	return output

### takes in states as the output of boat_simulation and plots the states -- using force###
def boat_sim_plot(X, show = True):
	### plot ###
	# u v r
	plt.figure()
	plt.subplot(311)
	plt.plot(X[-1,:], X[0,:])
	plt.ylabel('u [m/s]')
	plt.grid()

	plt.subplot(312)
	plt.plot(X[-1,:], X[1,:])
	plt.ylabel('v [m/s]')
	plt.grid()

	plt.subplot(313)
	plt.plot(X[-1,:], X[2,:])
	plt.ylabel('r [m/s]')
	plt.xlabel('Time [s]')
	plt.grid()

	### plot ###
	# du dv dr
	plt.figure()
	plt.subplot(311)
	plt.plot(X[-1,:], X[3,:])
	plt.ylabel('du [m/s²]')
	plt.grid()

	plt.subplot(312)
	plt.plot(X[-1,:], X[4,:])
	plt.ylabel('dv [m/s²]')
	plt.grid()

	plt.subplot(313)
	plt.plot(X[-1,:], X[5,:])
	plt.ylabel('dr [m/s²]')
	plt.xlabel('Time [s]')
	plt.grid()

	#inputs
	plt.figure()
	plt.subplot(311)
	plt.plot(X[-1,:],X[6,:])
	plt.ylabel('tau x [N]')
	plt.grid()

	plt.subplot(312)
	plt.plot(X[-1,:],X[7,:])
	plt.ylabel('tau y [N]')
	plt.grid()

	plt.subplot(313)
	plt.plot(X[-1,:],X[8,:])
	plt.ylabel('tau z [Nm]')
	plt.xlabel('time [s]')
	plt.grid()


	if show:
		plt.show()


### takes in states as the output of boat_simulation and plots the states -- using water jets  ###
def boat_sim_plot_wj(X, show = True):
	### plot ###
	# u v r
	plt.figure()
	plt.subplot(311)
	plt.plot(X[-1,:], X[0,:])
	plt.ylabel('u [m/s]')
	plt.grid()

	plt.subplot(312)
	plt.plot(X[-1,:], X[1,:])
	plt.ylabel('v [m/s]')
	plt.grid()

	plt.subplot(313)
	plt.plot(X[-1,:], X[2,:])
	plt.ylabel('r [m/s]')
	plt.xlabel('Time [s]')
	plt.grid()

	### plot ###
	# du dv dr
	plt.figure()
	plt.subplot(311)
	plt.plot(X[-1,:], X[3,:])
	plt.ylabel('du [m/s²]')
	plt.grid()

	plt.subplot(312)
	plt.plot(X[-1,:], X[4,:])
	plt.ylabel('dv [m/s²]')
	plt.grid()

	plt.subplot(313)
	plt.plot(X[-1,:], X[5,:])
	plt.ylabel('dr [m/s²]')
	plt.xlabel('Time [s]')
	plt.grid()

	#inputs
	plt.figure()
	plt.subplot(211)
	plt.plot(X[-1,:],X[6,:])
	plt.ylabel('noze angle [rad]')
	plt.grid()

	plt.subplot(212)
	plt.plot(X[-1,:],X[7,:])
	plt.ylabel('Jet RPM [N]')
	plt.xlabel('Time [s]')
	plt.grid()


	if show:
		plt.show()


#plots the tree structure -used in GPLEARN
def show_result(est_gp, X, Y, t, plot_show = False):
	graph = pydotplus.graphviz.graph_from_dot_data(est_gp._program.export_graphviz())
	img = Image.open(io.BytesIO(graph.create_png()))
	tree = np.asarray(img)
	plt.figure()
	plt.imshow(tree)

	print('RESULTING EQ: ',est_gp._program)
	Y_est = est_gp.predict(X)

	print('r²: ',est_gp.score(X,Y))
	plt.figure()
	plt.plot(t,Y)
	plt.plot(t,Y_est)
	plt.legend(['data', 'pred'])

	if plot_show:
		plt.show()


### --- jet model ---
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

def input_WJ(t, states):
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
		prev_jet_input.append(jet_rpm)
		prev_noz_input.append(delta_nozzle)
		return tau_b

	nu = states[3:6]
	jet_rpm = 500 
	if t < 18:
		delta_nozzle = 0
	elif t < 20:
		delta_nozzle = 0.2
		jet_rpm = 0
	else:
		delta_nozzle = 0
		jet_rpm = 100
	prev_t.append(t)
	tau_b = jet_model(nu, jet_rpm, delta_nozzle)


	return tau_b





####----- get acceleration data from txt file ------
"""
_IN_ 
path to the rosbag


_OUT_ 
x, y, z -acceleration, and valid_time, the time stamp for the acceleration data


"""
def acc_data(path_to_bag):
	path = '/home/gislehalv/Master/Data/NavData/'
	file0 = 'NavigationSolutionData-0-0000.txt'
	file1 = 'NavigationSolutionData-0-0001.txt'
	file2 = 'NavigationSolutionData-0-0002.txt'
	file3 = 'NavigationSolutionData-0-0003.txt'
	file4 = 'NavigationSolutionData-0-0004.txt'
	file5 = 'NavigationSolutionData-0-0005.txt'
	file6 = 'NavigationSolutionData-0-0006.txt'
	file7 = 'NavigationSolutionData-0-0007.txt'
	file8 = 'NavigationSolutionData-0-0008.txt'
	file9 = 'NavigationSolutionData-0-0009.txt'
	file10 = 'NavigationSolutionData-0-0010.txt'
	file11 = 'NavigationSolutionData-0-0011.txt'
	file12 = 'NavigationSolutionData-0-0012.txt'
	file13 = 'NavigationSolutionData-0-0013.txt'
	file14 = 'NavigationSolutionData-0-0014.txt'
	file15 = 'NavigationSolutionData-0-0015.txt'
	file16 = 'NavigationSolutionData-0-0016.txt'

	file_list = [file0, file1, file2, file3, file4, file5, file6, file7, file8, file9, file10, file11, file12, file13, file14, file15, file16]

	xvel = []
	yvel = []
	zvel = []
	#lat = []
	#long_ = []
	valid_time_vel = []
	#xvel = []
	for file in file_list:
		file_obj = open(path + file, 'r')

		i = 0
		for line in file_obj:

			if i > 40:
				#xvel.append(float(line.split(',')[6]))
				xvel.append(float(line.split(',')[6]))
				yvel.append(float(line.split(',')[7]))
				zvel.append(float(line.split(',')[14]))
				valid_time_vel.append(float(line.split(',')[30]) *1000) # to Nsec
				#lat.append(float(line.split(',')[0]))
				#long_.append(float(line.split(',')[1]))

			i = i+1


	###--test--
	path = '/home/gislehalv/Master/Data/NavData/'
	file0 = 'IMUInertialData-0-0000.txt'
	file1 = 'IMUInertialData-0-0001.txt'
	file2 = 'IMUInertialData-0-0002.txt'
	file3 = 'IMUInertialData-0-0003.txt'
	file4 = 'IMUInertialData-0-0004.txt'
	file5 = 'IMUInertialData-0-0005.txt'
	file6 = 'IMUInertialData-0-0006.txt'
	file7 = 'IMUInertialData-0-0007.txt'
	file8 = 'IMUInertialData-0-0008.txt'
	file9 = 'IMUInertialData-0-0009.txt'
	file10 = 'IMUInertialData-0-0010.txt'
	file11 = 'IMUInertialData-0-0011.txt'
	file12 = 'IMUInertialData-0-0012.txt'
	file13 = 'IMUInertialData-0-0013.txt'
	file14 = 'IMUInertialData-0-0014.txt'
	file15 = 'IMUInertialData-0-0015.txt'
	file16 = 'IMUInertialData-0-0016.txt'

	file_list = [file0, file1, file2, file3, file4, file5, file6, file7, file8, file9, file10, file11, file12, file13, file14, file15, file16]
	
	xacc = []
	yacc = []
	zacc = []
	#lat = []
	#long_ = []
	valid_time = []
	#xvel = []
	for file in file_list:
		file_obj = open(path + file, 'r')

		i = 0
		for line in file_obj:

			if i > 19:
				#xvel.append(float(line.split(',')[6]))
				xacc.append(float(line.split(',')[6]))
				yacc.append(float(line.split(',')[7]))
				zacc.append(float(line.split(',')[5]))
				valid_time.append(float(line.split(',')[9]) *1000) # to Nsec
				#lat.append(float(line.split(',')[0]))
				#long_.append(float(line.split(',')[1]))

			i = i+1

	bag = rosbag.Bag(path_to_bag)
	bagContents = bag.read_messages()

	# #### -- jet data --
	# jet_data  = [] 
	# for i, subtopic, msg, t in enumerate(bag.read_messages('/usv/hamjet/status_high')):
	# 	jet_data[2, i].append(float(str(t)))
	
	#### -- nav data --
	# cnt_jet = 0
	# for subtopic, msg, t in bag.read_messages('/usv/hamjet/status_high'):
	# 	cnt_jet += 1
	# test_time = []
	# jet_data  = np.zeros((4,cnt_jet))
	# i = 0
	nav_data = []
	for subtopic, msg, t in bag.read_messages('/usv/navp_msg'):
		nav_data.append(float(str(t)))
		
	#--set init time to zero
	nav_data = np.divide(np.subtract(nav_data, valid_time[0]), 1e9) # to sec
	#jet_data[2, :]  =  np.divide(np.subtract(jet_data[2,:], valid_time[0]), 1e9)# to sec
	valid_time = np.divide(np.subtract(valid_time, valid_time[0]), 1e9) # to sec
	valid_time_vel = np.divide(np.subtract(valid_time_vel, valid_time_vel[0]), 1e9)


	start_vel = np.argmin((valid_time_vel - nav_data[0])**2)
	stop_vel = np.argmin((valid_time_vel - nav_data[-1])**2)

	# about 7.8 sec diff
	#nav_data = np.add(nav_data, 7.8)
	
	

	start = np.argmin((valid_time - nav_data[0])**2)
	stop = np.argmin((valid_time - nav_data[-1])**2)






	#test if it's a good match 
	if np.abs(valid_time[start] - nav_data[0]) > 0.5 or np.abs(valid_time[stop] - nav_data[-1]) > 0.5:
		print('the arrays do not match')
		exit()
	
	return xacc[start:stop], yacc[start:stop], zacc[start:stop], valid_time[start:stop], xvel[start_vel:stop_vel], yvel[start_vel:stop_vel], zvel[start_vel:stop_vel], valid_time_vel[start_vel:stop_vel]








"""  ------OPEN BAG-------- 
Opens, filtes and interpolates a bag
_IN_ 
path:	path to bag file
plot: 	whether or not to plot the contet, default = False 
thr_bucket: cuts the outputs to only contain data with bucket > 95. check data to see if its necessary
filter_cutoff:  what cutoff frq of the butterworth filter to use. shuld be in the range [0.01, 0.1]


_OUT_ 
X: 		[u_smooth, v_smooth, r_smooth, du_smo, dv_smo, dr_smo, jet_rpm, nozzle_angle, bucket, interp_arr]
"""



def open_bag(path, plot = False, thr_bucket = True, filter_cutoff = 0.025, return_raw_data = False):
	bag = rosbag.Bag(path)
	bagContents = bag.read_messages()
	if not bagContents:
		print('bag is empty')
		exit()

	#save?
	save = False
	
	#### -- jet data --
	cnt_jet = 0
	for subtopic, msg, t in bag.read_messages('/usv/hamjet/status_high'):
		cnt_jet += 1

	jet_data  = np.zeros((4,cnt_jet))
	i = 0
	for subtopic, msg, t in bag.read_messages('/usv/hamjet/status_high'):
		# engines rpm, steering, jet time, bucket
		jet_data[0, i] = (msg.port_shaft_rpm + msg.stbd_shaft_rpm) / 2
		jet_data[1, i] = (msg.port_steering + msg.stbd_steering) / 2
		jet_data[2, i] = t.to_sec()
		jet_data[3, i] = (msg.port_reverse + msg.stbd_reverse) / 2
		i += 1

	# Set initail jet time to zero
	jet_data[2, :] = jet_data[2, :] - jet_data[2, 0]


	#### -- nav data --
	cnt_nav = 0
	for subtopic, msg, t in bag.read_messages('/usv/navp_msg'):
		cnt_nav += 1

	nav_data = np.zeros((7,cnt_nav))
	i = 0
	for subtopic, msg, t in bag.read_messages('/usv/navp_msg'):
		#surge, sway, yaw
		nav_data[0, i] = msg.pose.latitude
		nav_data[1, i] = msg.pose.longitude
		nav_data[2, i] = msg.pose.heading

		#surge, sway, yaw - Rate
		nav_data[3, i] = msg.vel.xVelocityB
		nav_data[4, i] = msg.vel.yVelocityB
		nav_data[5, i] = msg.rate.zAngularRateB #prob in deg/s

		# nav time
		nav_data[6, i] = t.to_sec()
		i += 1

	#set initial time to zero
	nav_data[6, :] = nav_data[6, :] - nav_data[6, 0] 
	nav_data[5, :] = nav_data[5, :] * (np.pi/180) #from deg/s to rad/s



	####		 --Fiter--
	#set the sample time to 0.05
	
	def filter(nav_data):
		#savgol_filter
		sav_fil = False
		FB_fil = False
		spline = False
		interpol = True

		#savgol_filter
		if sav_fil:
			# u_smooth_deriv = signal.savgol_filter(nav_data[3, :], 51, 4, deriv=1, delta=nav_data[3, 1]- nav_data[3, 0]) 
			# v_smooth_deriv = signal.savgol_filter(nav_data[4, :], 51, 4, deriv=1, delta=nav_data[3, 1]- nav_data[3, 0]) 
			# r_smooth_deriv = signal.savgol_filter(nav_data[5, :], 51, 4, deriv=1, delta=nav_data[3, 1]- nav_data[3, 0]) 

			u_smooth = signal.savgol_filter(nav_data[3, :], 51, 4) 
			v_smooth = signal.savgol_filter(nav_data[4, :], 51, 4) 
			r_smooth = signal.savgol_filter(nav_data[5, :], 51, 4) 

		#find freq
		if 0:
			len_ = 30
			test_t = list(np.arange(0, len_, 0.01))
			sp = np.fft.rfft(np.sin(test_t))
			freq = np.fft.rfftfreq(len(test_t))
			plt.figure()
			plt.plot(freq, sp.real)
			plt.grid()
			plt.ylabel('amount?')
			plt.xlabel('freq')

			#butter
			order = 2
			cutoff = 0.003
			b, a = signal.butter(order, cutoff, btype='low', analog=False, output='ba')

			#Forward -Backward filter
			if FB_fil:
				sin = signal.filtfilt(b, a, np.sin(test_t)) 
		 

			plt.figure()
			plt.plot(test_t, sin)
			plt.plot(test_t, np.sin(test_t))
			plt.legend(['butter', 'orig'])

			plt.show()
			exit()
		
		if FB_fil:
			#butter
			order = 2
			cutoff = 0.05
			b, a = signal.butter(order, cutoff, btype='low', analog=False, output='ba')

			#Forward -Backward filter
			u_smooth = signal.filtfilt(b, a, nav_data[3, :]) 
			v_smooth = signal.filtfilt(b, a, nav_data[4, :]) 
			r_smooth = signal.filtfilt(b, a, nav_data[5, :]) 

			#forward
			# u_smooth = signal.lfilter(b, a, nav_data[3, :]) 
			# v_smooth = signal.lfilter(b, a, nav_data[4, :]) 
			# r_smooth = signal.lfilter(b, a, nav_data[5, :]) 
		
		#spline 
		if spline: #NOPE
			u_smooth = signal.gauss_spline(nav_data[3, :], 5) 
			v_smooth = signal.gauss_spline(nav_data[4, :], 5) 
			r_smooth = signal.gauss_spline(nav_data[5, :], 5) 

		#interpolate then smooth
		if interpol:
			steps = 0.05
			interp_arr = list(np.arange(nav_data[6, 0], nav_data[6, -1], steps))
			u_int = np.interp(interp_arr, nav_data[6, :],nav_data[3, :])
			v_int = np.interp(interp_arr, nav_data[6, :],nav_data[4, :])
			r_int = np.interp(interp_arr, nav_data[6, :],nav_data[5, :])

			order = 2
			cutoff = filter_cutoff
			b, a = signal.butter(order, cutoff, btype='low', analog=False, output='ba')

			#Forward -Backward filter
			u_smooth = signal.filtfilt(b, a, u_int) 
			v_smooth = signal.filtfilt(b, a, v_int) 
			r_smooth = signal.filtfilt(b, a, r_int) 
			return u_smooth, v_smooth, r_smooth, interp_arr
			#plot
			if 0:
				plt.figure()
				plt.plot(nav_data[6, :], nav_data[3, :])
				plt.plot(interp_arr, u_int)
				plt.plot(interp_arr, u_smooth)
				plt.legend([ 'orig', 'int', 'int smooth'])
				plt.ylabel('u')
				plt.grid()

				plt.figure()
				plt.plot(nav_data[6, :], nav_data[4, :])
				plt.plot(interp_arr, v_int)
				plt.plot(interp_arr, v_smooth)
				plt.legend([ 'orig', 'int', 'int smooth'])
				plt.ylabel('v')
				plt.grid()

				plt.figure()
				plt.plot(nav_data[6, :], nav_data[5, :])
				plt.plot(interp_arr, r_int)
				plt.plot(interp_arr, r_smooth)
				plt.legend([ 'orig', 'int', 'int smooth'])
				plt.ylabel('r')
				plt.grid()

				plt.show()
		
		return u_smooth, v_smooth, r_smooth
	u_smooth, v_smooth, r_smooth, interp_arr = filter(nav_data)


	#### 		-- Integrate --
	def deriv(nav_data, u_smooth, v_smooth, r_smooth):
		du = np.diff(nav_data[3, :],n = 1) / 0.05
		dv = np.diff(nav_data[4, :],n = 1)/ 0.05
		dr = np.diff(nav_data[5, :],n = 1)/ 0.05

		du_smo = np.diff(u_smooth ,n = 1)/ 0.05
		dv_smo = np.diff(v_smooth ,n = 1)/ 0.05
		dr_smo = np.diff(r_smooth ,n = 1)/ 0.05

		#add the last signal twice 
		du = np.concatenate((du,[du[-1]]))
		dv = np.concatenate((dv,[dv[-1]]))
		dr = np.concatenate((dr,[dr[-1]]))

		du_smo = np.concatenate((du_smo,[du_smo[-1]]))
		dv_smo = np.concatenate((dv_smo,[dv_smo[-1]]))
		dr_smo = np.concatenate((dr_smo,[dr_smo[-1]]))

		return du, dv, dr, du_smo, dv_smo, dr_smo
	du, dv, dr, du_smo, dv_smo, dr_smo =  deriv(nav_data, u_smooth, v_smooth, r_smooth)

	jet_data[1, :] = np.multiply(jet_data[1, :],(27/100)) # from [-100, 100] to [-27, 27] deg

	def interpolate(jet_data, interp_arr):
		jet_rpm = 		np.interp(interp_arr, jet_data[2, :],jet_data[0, :])
		nozzle_angle =	np.interp(interp_arr, jet_data[2, :],jet_data[1, :]) # not really nozzle angle but rather [-100, 100]% = [-27, 27] deg 
		bucket = 		np.interp(interp_arr, jet_data[2, :],jet_data[3, :]) # from [-100, to 100]
		return jet_rpm, nozzle_angle, bucket
	jet_rpm, nozzle_angle, bucket = interpolate(jet_data, interp_arr)



	


	#preeprossesed matrix of variables.
	X = [u_smooth, v_smooth, r_smooth, du_smo, dv_smo, dr_smo, jet_rpm, nozzle_angle, bucket, interp_arr]
	X = np.array(X)



	# a test to check if the bucket is fully open in all the data. if not - start where it becomes > 95 and end if it <95
	def bucket_fully_open(X):
		if X[-2, 0] > 95:
			start = 0
		else:
			start = -1
		for i in range(np.shape(X)[1]):
			if X[-2, i] > 95 and start == -1:
				start = i
			if X[-2, i] < 95 and start == -1:
				continue 
			if X[-2, i] < 95 and start != -1:
				stop = i
				break

		X = X[:,start:stop]
		return X
	
	if thr_bucket: #use the  bucket_fully_open function 
		X = bucket_fully_open(X)

	### ---Plots----
	if plot:
		plt.figure()
		plt.plot(jet_data[2, :], jet_data[0, :])
		plt.plot(jet_data[2, :], jet_data[1, :])
		plt.plot(jet_data[2, :], jet_data[3, :])
		plt.legend(['rpm','steering','bucket'])
		plt.grid()

		plt.figure()
		plt.subplot(311)
		plt.plot(nav_data[6, :], du)
		plt.plot(interp_arr, du_smo)
		plt.legend(['du', 'du_smooth'])
		plt.grid()
		plt.subplot(312)
		plt.plot(nav_data[6, :],dv)
		plt.plot(interp_arr,dv_smo)
		plt.legend(['dv', 'dv_smooth'])
		plt.grid()
		plt.subplot(313)
		plt.plot(nav_data[6, :],dr)
		plt.plot(interp_arr,dr_smo)
		plt.legend(['dr', 'dr_smooth'])
		plt.grid()

		plt.figure()
		plt.subplot(311)
		plt.plot(nav_data[6, :], nav_data[3, :], 'r.-')
		plt.plot(interp_arr, u_smooth)
		plt.ylabel('u')
		plt.grid()
		plt.subplot(312)
		plt.plot(nav_data[6, :], nav_data[4, :], 'r.-')
		plt.plot(interp_arr, v_smooth)
		plt.ylabel('v')
		plt.grid()
		plt.subplot(313)
		plt.plot(nav_data[6, :], nav_data[5, :], 'r.-')
		plt.plot(interp_arr, r_smooth)
		plt.ylabel('r')
		plt.grid()

		plt.figure()
		plt.subplot(211)
		plt.plot(jet_data[2, :], jet_data[1, :])
		plt.grid()
		plt.ylabel('nozzle')
		plt.subplot(212)
		plt.plot(nav_data[6, :], nav_data[4, :])
		plt.grid()
		plt.ylabel('v')

		plt.figure()
		plt.plot(nav_data[0,:],nav_data[1,:])
		plt.plot(nav_data[0,0],nav_data[1,0],'rx')
		plt.title('XY-plot, starts at the red cross')

		plt.figure()
		plt.subplot(311)
		plt.plot(jet_data[2],jet_data[0])
		plt.ylabel('Jet [RPM]')
		plt.grid()
		plt.subplot(312)
		plt.plot(nav_data[6],nav_data[3])
		plt.plot(interp_arr, u_smooth)
		plt.legend(['raw', 'smooth'])
		plt.ylabel('u [m/s]')
		plt.grid()
		plt.subplot(313)
		plt.plot(nav_data[6],du)	
		plt.plot(interp_arr,du_smo)
		plt.legend(['raw', 'smooth'])
		plt.ylabel('du smooth [m/s^2]')
		plt.xlabel('Time [s]')
		plt.grid()


		plt.figure()
		plt.subplot(311)
		plt.plot(jet_data[2],jet_data[1])
		plt.ylabel('nozzle angle')
		plt.grid()
		plt.subplot(312)
		plt.plot(nav_data[6],nav_data[4])
		plt.plot(interp_arr, v_smooth)
		plt.legend(['raw', 'smooth'])
		plt.ylabel('v [m/s]')
		plt.grid()
		plt.subplot(313)
		plt.plot(nav_data[6],dv)	
		plt.plot(interp_arr,dv_smo)
		plt.legend(['raw', 'smooth'])
		plt.ylabel('dv smooth [m/s^2]')
		plt.xlabel('Time [s]')
		plt.grid()


		plt.figure()
		plt.plot(nav_data[6, :], du)
		plt.plot(interp_arr, du_smo)
		plt.ylabel('du [m/s^2]')
		plt.xlabel('Time [s]')
		plt.legend(['du - original', 'du - after butterworth'])
		plt.grid()

		plt.show()

	### --- save -- does not work any more
	if save:
		##  - interpolate --
		#interpolate to jet time
		data = np.zeros((10,len(jet_data[2, :])))
		data[0, :] = np.interp(jet_data[2,:], nav_data[6, :],u_smooth)
		data[1, :] = np.interp(jet_data[2,:], nav_data[6, :],v_smooth)
		data[2, :] = np.interp(jet_data[2,:], nav_data[6, :],r_smooth)
		data[3, :] = np.interp(jet_data[2,:], nav_data[6, :],du_smo)
		data[4, :] = np.interp(jet_data[2,:], nav_data[6, :],dv_smo)
		data[5, :] = np.interp(jet_data[2,:], nav_data[6, :],dr_smo)

		data[6, :] = jet_data[0, :] # engine 
		data[7, :] = jet_data[1, :] # steering
		data[8, :] = jet_data[3, :] # bucket
		data[9, :] = jet_data[2, :] # jet time


		save_path = '/home/gislehalv/Master/Data/CSV Data From Bags/'
		save_name = save_path + name + '.csv'
		np.savetxt(save_name, data, delimiter = ',')

	if return_raw_data:
		#interpolate to jet time
		u_jet_t = np.interp(jet_data[2, :], nav_data[6, i], nav_data[3, :])
		v_jet_t = np.interp(jet_data[2, :], nav_data[6, i], nav_data[4, :])
		r_jet_t = np.interp(jet_data[2, :], nav_data[6, i], nav_data[5, :])
		du_jet_t = np.interp(jet_data[2, :], nav_data[6, i], du)
		dv_jet_t = np.interp(jet_data[2, :], nav_data[6, i], dv)
		dr_jet_t = np.interp(jet_data[2, :], nav_data[6, i], dr)

		X = [u_jet_t, v_jet_t, r_jet_t, du_jet_t, dv_jet_t, dr_jet_t, jet_rpm, nozzle_angle, bucket, jet_data[2, :]]
		return X

	return X





def open_bag_w_yaw(path, plot = False, thr_bucket = True, filter_cutoff = 0.025, return_raw_data = False):
	bag = rosbag.Bag(path)
	bagContents = bag.read_messages()
	if not bagContents:
		print('bag is empty')
		exit()

	#### -- jet data --
	cnt_jet = 0
	for subtopic, msg, t in bag.read_messages('/usv/hamjet/status_high'):
		cnt_jet += 1

	jet_data  = np.zeros((4,cnt_jet))
	i = 0
	for subtopic, msg, t in bag.read_messages('/usv/hamjet/status_high'):
		# engines rpm, steering, jet time, bucket
		jet_data[0, i] = (msg.port_shaft_rpm + msg.stbd_shaft_rpm) / 2
		jet_data[1, i] = (msg.port_steering + msg.stbd_steering) / 2
		jet_data[2, i] = t.to_sec()
		jet_data[3, i] = (msg.port_reverse + msg.stbd_reverse) / 2
		i += 1


	# Set initail jet time to zero
	jet_data[2, :] = jet_data[2, :] - jet_data[2, 0]


	#### -- nav data --
	cnt_nav = 0
	for subtopic, msg, t in bag.read_messages('/usv/navp_msg'):
		cnt_nav += 1

	nav_data = np.zeros((7,cnt_nav))
	i = 0
	for subtopic, msg, t in bag.read_messages('/usv/navp_msg'):
		#surge, sway, yaw
		nav_data[0, i] = msg.pose.latitude
		nav_data[1, i] = msg.pose.longitude
		nav_data[2, i] = msg.pose.heading

		#surge, sway, yaw - Rate
		nav_data[3, i] = msg.vel.xVelocityB
		nav_data[4, i] = msg.vel.yVelocityB
		nav_data[5, i] = msg.rate.zAngularRateB # in deg/s

		# nav time
		nav_data[6, i] = t.to_sec()
		i += 1

	#set initial time to zero
	nav_data[6, :] = nav_data[6, :] - nav_data[6, 0] 
	nav_data[5, :] = nav_data[5, :] * (np.pi/180) #from deg/s to rad/s
	####		 --Fiter--
	#set the sample time to 0.05
	def filter(nav_data):
		#savgol_filter
		sav_fil = False
		FB_fil = False
		spline = False
		interpol = True

		#savgol_filter
		if sav_fil:
			# u_smooth_deriv = signal.savgol_filter(nav_data[3, :], 51, 4, deriv=1, delta=nav_data[3, 1]- nav_data[3, 0]) 
			# v_smooth_deriv = signal.savgol_filter(nav_data[4, :], 51, 4, deriv=1, delta=nav_data[3, 1]- nav_data[3, 0]) 
			# r_smooth_deriv = signal.savgol_filter(nav_data[5, :], 51, 4, deriv=1, delta=nav_data[3, 1]- nav_data[3, 0]) 

			u_smooth = signal.savgol_filter(nav_data[3, :], 51, 4) 
			v_smooth = signal.savgol_filter(nav_data[4, :], 51, 4) 
			r_smooth = signal.savgol_filter(nav_data[5, :], 51, 4) 

		#find freq
		if 0:
			len_ = 30
			test_t = list(np.arange(0, len_, 0.01))
			sp = np.fft.rfft(np.sin(test_t))
			freq = np.fft.rfftfreq(len(test_t))
			plt.figure()
			plt.plot(freq, sp.real)
			plt.grid()
			plt.ylabel('amount?')
			plt.xlabel('freq')

			#butter
			order = 2
			cutoff = 0.003
			b, a = signal.butter(order, cutoff, btype='low', analog=False, output='ba')

			#Forward -Backward filter
			if FB_fil:
				sin = signal.filtfilt(b, a, np.sin(test_t)) 
		 

			plt.figure()
			plt.plot(test_t, sin)
			plt.plot(test_t, np.sin(test_t))
			plt.legend(['butter', 'orig'])

			plt.show()
			exit()
		
		if FB_fil:
			#butter
			order = 2
			cutoff = 0.05
			b, a = signal.butter(order, cutoff, btype='low', analog=False, output='ba')

			#Forward -Backward filter
			u_smooth = signal.filtfilt(b, a, nav_data[3, :]) 
			v_smooth = signal.filtfilt(b, a, nav_data[4, :]) 
			r_smooth = signal.filtfilt(b, a, nav_data[5, :]) 

			#forward
			# u_smooth = signal.lfilter(b, a, nav_data[3, :]) 
			# v_smooth = signal.lfilter(b, a, nav_data[4, :]) 
			# r_smooth = signal.lfilter(b, a, nav_data[5, :]) 
		
		#spline 
		if spline: #NOPE
			u_smooth = signal.gauss_spline(nav_data[3, :], 5) 
			v_smooth = signal.gauss_spline(nav_data[4, :], 5) 
			r_smooth = signal.gauss_spline(nav_data[5, :], 5) 

		#interpolate then smooth
		if interpol:
			steps = 0.05
			interp_arr = list(np.arange(nav_data[6, 0], nav_data[6, -1], steps))
			u_int = np.interp(interp_arr, nav_data[6, :],nav_data[3, :])
			v_int = np.interp(interp_arr, nav_data[6, :],nav_data[4, :])
			r_int = np.interp(interp_arr, nav_data[6, :],nav_data[5, :])

			order = 2
			cutoff = filter_cutoff
			b, a = signal.butter(order, cutoff, btype='low', analog=False, output='ba')

			#Forward -Backward filter
			u_smooth = signal.filtfilt(b, a, u_int) 
			v_smooth = signal.filtfilt(b, a, v_int) 
			r_smooth = signal.filtfilt(b, a, r_int) 
			return u_smooth, v_smooth, r_smooth, interp_arr
			#plot
			if 0:
				plt.figure()
				plt.plot(nav_data[6, :], nav_data[3, :])
				plt.plot(interp_arr, u_int)
				plt.plot(interp_arr, u_smooth)
				plt.legend([ 'orig', 'int', 'int smooth'])
				plt.ylabel('u')
				plt.grid()

				plt.figure()
				plt.plot(nav_data[6, :], nav_data[4, :])
				plt.plot(interp_arr, v_int)
				plt.plot(interp_arr, v_smooth)
				plt.legend([ 'orig', 'int', 'int smooth'])
				plt.ylabel('v')
				plt.grid()

				plt.figure()
				plt.plot(nav_data[6, :], nav_data[5, :])
				plt.plot(interp_arr, r_int)
				plt.plot(interp_arr, r_smooth)
				plt.legend([ 'orig', 'int', 'int smooth'])
				plt.ylabel('r')
				plt.grid()

				plt.show()
		
		return u_smooth, v_smooth, r_smooth
	u_smooth, v_smooth, r_smooth, interp_arr = filter(nav_data)


	#### 		-- Integrate --
	def deriv(nav_data, u_smooth, v_smooth, r_smooth):
		du = np.diff(nav_data[3, :],n = 1) / 0.05
		dv = np.diff(nav_data[4, :],n = 1)/ 0.05
		dr = np.diff(nav_data[5, :],n = 1)/ 0.05

		du_smo = np.diff(u_smooth ,n = 1)/ 0.05
		dv_smo = np.diff(v_smooth ,n = 1)/ 0.05
		dr_smo = np.diff(r_smooth ,n = 1)/ 0.05

		#add the last signal twice 
		du = np.concatenate((du,[du[-1]]))
		dv = np.concatenate((dv,[dv[-1]]))
		dr = np.concatenate((dr,[dr[-1]]))

		du_smo = np.concatenate((du_smo,[du_smo[-1]]))
		dv_smo = np.concatenate((dv_smo,[dv_smo[-1]]))
		dr_smo = np.concatenate((dr_smo,[dr_smo[-1]]))

		return du, dv, dr, du_smo, dv_smo, dr_smo
	du, dv, dr, du_smo, dv_smo, dr_smo =  deriv(nav_data, u_smooth, v_smooth, r_smooth)

	jet_data[1, :] = np.multiply(jet_data[1, :],(27/100)) # from [-100, 100] to [-27, 27] deg

	def interpolate(jet_data, nav_data, interp_arr):
		jet_rpm = 		np.interp(interp_arr, jet_data[2, :], jet_data[0, :])
		nozzle_angle =	np.interp(interp_arr, jet_data[2, :], jet_data[1, :]) # not really nozzle angle but rather [-100, 100]% = [-27, 27] deg 
		bucket = 		np.interp(interp_arr, jet_data[2, :], jet_data[3, :]) # from [-100, to 100]
		yaw = 			np.interp(interp_arr, nav_data[6, :], nav_data[2, :]) # yaw
		return jet_rpm, nozzle_angle, bucket, yaw

	jet_rpm, nozzle_angle, bucket, yaw = interpolate(jet_data, nav_data, interp_arr)



	


	#preeprossesed matrix of variables.
	X = [yaw, u_smooth, v_smooth, r_smooth, du_smo, dv_smo, dr_smo, jet_rpm, nozzle_angle, bucket, interp_arr]
	X = np.array(X)


	### ---Plots----
	if plot:
		plt.figure()
		plt.plot(jet_data[2, :], jet_data[0, :])
		plt.plot(jet_data[2, :], jet_data[1, :])
		plt.plot(jet_data[2, :], jet_data[3, :])
		plt.legend(['rpm','steering','bucket'])
		plt.grid()

		plt.figure()
		plt.subplot(311)
		plt.plot(nav_data[6, :], du)
		plt.plot(interp_arr, du_smo)
		plt.legend(['du', 'du_smooth'])
		plt.grid()
		plt.subplot(312)
		plt.plot(nav_data[6, :],dv)
		plt.plot(interp_arr,dv_smo)
		plt.legend(['dv', 'dv_smooth'])
		plt.grid()
		plt.subplot(313)
		plt.plot(nav_data[6, :],dr)
		plt.plot(interp_arr,dr_smo)
		plt.legend(['dr', 'dr_smooth'])
		plt.grid()

		plt.figure()
		plt.subplot(311)
		plt.plot(nav_data[6, :], nav_data[3, :], 'r.-')
		plt.plot(interp_arr, u_smooth)
		plt.ylabel('u')
		plt.grid()
		plt.subplot(312)
		plt.plot(nav_data[6, :], nav_data[4, :], 'r.-')
		plt.plot(interp_arr, v_smooth)
		plt.ylabel('v')
		plt.grid()
		plt.subplot(313)
		plt.plot(nav_data[6, :], nav_data[5, :], 'r.-')
		plt.plot(interp_arr, r_smooth)
		plt.ylabel('r')
		plt.grid()

		plt.figure()
		plt.subplot(211)
		plt.plot(jet_data[2, :], jet_data[1, :])
		plt.grid()
		plt.ylabel('nozzle')
		plt.subplot(212)
		plt.plot(nav_data[6, :], nav_data[4, :])
		plt.grid()
		plt.ylabel('v')

		plt.figure()
		plt.plot(nav_data[0,:],nav_data[1,:])
		plt.plot(nav_data[0,0],nav_data[1,0],'rx')
		plt.title('XY-plot, starts at the red cross')

		plt.figure()
		plt.subplot(311)
		plt.plot(jet_data[2],jet_data[0])
		plt.ylabel('Jet [RPM]')
		plt.grid()
		plt.subplot(312)
		plt.plot(nav_data[6],nav_data[3])
		plt.plot(interp_arr, u_smooth)
		plt.legend(['raw', 'smooth'])
		plt.ylabel('u [m/s]')
		plt.grid()
		plt.subplot(313)
		plt.plot(nav_data[6],du)	
		plt.plot(interp_arr,du_smo)
		plt.legend(['raw', 'smooth'])
		plt.ylabel('du smooth [m/s^2]')
		plt.xlabel('Time [s]')
		plt.grid()


		plt.figure()
		plt.subplot(311)
		plt.plot(jet_data[2],jet_data[1])
		plt.ylabel('nozzle angle')
		plt.grid()
		plt.subplot(312)
		plt.plot(nav_data[6],nav_data[4])
		plt.plot(interp_arr, v_smooth)
		plt.legend(['raw', 'smooth'])
		plt.ylabel('v [m/s]')
		plt.grid()
		plt.subplot(313)
		plt.plot(nav_data[6],dv)	
		plt.plot(interp_arr,dv_smo)
		plt.legend(['raw', 'smooth'])
		plt.ylabel('dv smooth [m/s^2]')
		plt.xlabel('Time [s]')
		plt.grid()

		plt.show()

	return X




