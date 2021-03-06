"""
Simulate the boat with real data

"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/gislehalv/Master/scripts')
import my_lib
from scipy.optimize import least_squares 
from pytictoc import TicToc

###-- get data
# X1 = np.load('/home/gislehalv/Master/Data/numpy_data_from_bag/' + 'bag1_025'+'.npy')
# X2 = np.load('/home/gislehalv/Master/Data/numpy_data_from_bag/' + 'bag2_025'+'.npy')
# X3 = np.load('/home/gislehalv/Master/Data/numpy_data_from_bag/' + 'bag3_025'+'.npy')
# X4 = np.load('/home/gislehalv/Master/Data/numpy_data_from_bag/' + 'bag4_025'+'.npy')


# #fix time 
# X2[-1] = X2[-1] + X1[-1, -1]
# X3[-1] = X3[-1] + X2[-1, -1]
# X4[-1] = X4[-1] + X3[-1, -1]

# #melt all the data together
# X = np.concatenate((X1,X2,X3,X4),axis = 1)

# index = []
# for i in range(np.shape(X)[1]):
# 	if np.shape(X)[1] > i:
# 		if X[-2, i] < 95:
# 			index.append(i)

# X = np.delete(X, index, 1)

bag_1 = 'hal_control_2018-12-11-10-53-26_0' #large!
bag_2 = 'hal_control_2018-12-11-11-49-22_0' #similar to bag1 but smaller
bag_3 = 'hal_control_2018-12-11-12-13-58_0' #
bag_4 = 'hal_control_2018-12-11-12-13-58_0'



# bag path
path = '/home/gislehalv/Master/Data/'

# bagFile_path_train = path + bag_4 + '.bag'

# bagFile_path_val = path + bag_1 + '.bag'

bagFile1 = path + bag_1 + '.bag'
bagFile2 = path + bag_2 + '.bag'
bagFile3 = path + bag_3 + '.bag'
bagFile4 = path + bag_4 + '.bag'

# get data
X1 = my_lib.open_bag_w_yaw(bagFile1, plot=False, thr_bucket = True, filter_cutoff = 0.025)
X2 = my_lib.open_bag_w_yaw(bagFile2, plot=False, thr_bucket = True, filter_cutoff = 0.025)
X3 = my_lib.open_bag_w_yaw(bagFile3, plot=False, thr_bucket = True, filter_cutoff = 0.025)
X4 = my_lib.open_bag_w_yaw(bagFile4, plot=False, thr_bucket = True, filter_cutoff = 0.025)

X = np.concatenate((X1,X2,X3,X4),axis = 1)

X[0,:] = np.multiply(X[0,:], np.pi/180) #deg2rad
X[-3,:] = np.multiply(X[-3,:], np.pi/180) #deg2rad



##WJ model - from jet input to force
def input_WJ(states):
	def jet_model(nu, jet_rpm, delta_nozzle):

		#constants
		lever_CGtowj_port = [-3.82, -0.475]
		lever_CGtowj_stb = [-3.82, 0.475]
		#rpm_slew_rate = [2000, -2000]
		#nozzle_slew_rate = [1.3464, -1.3464]
		#rpm_min_max = [0, 2000]

		# if jet_rpm > rpm_min_max[1]:
		# 	jet_rpm = rpm_min_max[1]
		# elif jet_rpm < rpm_min_max[0]:
		# 	jet_rpm = rpm_min_max[0]

		
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

		thrust_list.append(thrust)
		#waterjet port
		#force
		Fx = thrust*np.cos(delta_nozzle)
		Fy = thrust*np.sin(delta_nozzle)
		#moment
		Nz_port = (lever_CGtowj_port[0]*Fy)- (lever_CGtowj_port[1]*Fx)
		Nz_stb = (lever_CGtowj_stb[0]*Fy)- (lever_CGtowj_stb[1]*Fx)

		tau_b =  [2*Fx, 2*Fy, Nz_port + Nz_stb]

		return tau_b

	nu = states[1:4]
	jet_rpm = states[-4]
	delta_nozzle = states[-3] #* (np.pi/180) #deg to rad

	tau_b = jet_model(nu, jet_rpm, delta_nozzle)

	return tau_b


## System ###
def sys(states,WaterJetModel):

	tauB = np.zeros((3,1))
	tauB[:,0] = WaterJetModel(states)

	#save force
	tauB_array[:, i] = np.squeeze(tauB)

	
	nu = np.zeros((3,1))
	nu[0,0] = states[1]
	nu[1,0] = states[2]
	nu[2,0] = states[3]

	yaw = states[0];


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


# tauB_array = np.zeros((3, np.shape(X[-1])[0]))
# thrust_list = []
# out = np.zeros((6, np.shape(X)[1]))


# for i in range(np.shape(X)[1]):
# 	out[:,i] = np.squeeze(sys(X[:,i], input_WJ))




###--- Curve fit the params of the WJ model --
#force_list = []

def input_WJ_w_params(states, params):
	def jet_model(nu, jet_rpm, delta_nozzle, params):
		a0, a1, a2, R, r0, r1, r2 = params

		#constants
		lever_CGtowj_port = [-3.82, -0.475]
		lever_CGtowj_stb = [-3.82, 0.475]

		#rpm2thrust
		speed = nu[0] * 1.94384 # knots 
		# a0 = 6244.15
		# a1 = -178.46
		# a2 = 0.881043
		thrust_unscaled = a0 + a1*speed + a2*(speed**2)

		# r0 = 85.8316
		# r1 = -1.7935
		# r2 = 0.00533
		#R = 1/4530
		rpm_scale = R*(r0 + r1*jet_rpm + r2 * (jet_rpm **2)) 

		thrust = rpm_scale * thrust_unscaled

		#thrust_list.append(thrust)
		#waterjet port
		#force
		Fx = thrust*np.cos(delta_nozzle)
		Fy = thrust*np.sin(delta_nozzle)
		#moment
		Nz_port = (lever_CGtowj_port[0]*Fy)- (lever_CGtowj_port[1]*Fx)
		Nz_stb = (lever_CGtowj_stb[0]*Fy)- (lever_CGtowj_stb[1]*Fx)

		tau_b =  [2*Fx, 2*Fy, Nz_port + Nz_stb]

		return tau_b

	nu = states[1:4]
	jet_rpm = states[-4]
	delta_nozzle = states[-3] #* (np.pi/180) #deg to rad

	tau_b = jet_model(nu, jet_rpm, delta_nozzle, params)
	#force_list.append(list(tau_b))
	return tau_b

def sys_w_params(states,WaterJetModel, params):

	tauB = np.zeros((3,1))
	tauB[:,0] = WaterJetModel(states, params)

	#save force
	#tauB_array[:, i] = np.squeeze(tauB)

	
	nu = np.zeros((3,1))
	nu[0,0] = states[1]
	nu[1,0] = states[2]
	nu[2,0] = states[3]

	yaw = states[0];


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

def optimize_params(X):
	def tot_sys(x):
		out = np.zeros((6, np.shape(X)[1]))
		for i in range(np.shape(X)[1]):
			out[:,i] = np.squeeze(sys_w_params(X[:,i], input_WJ_w_params, x))
		se_u = (X[4] - out[3])**2
		se_v = (X[5] - out[4])**2
		se_r = (X[6] - out[5])**2
		se = se_u + se_v + se_r 
		mse = np.sum(se)/len(se)
		print(mse)
		return se

	#x0 = np.ones(7)
	x0 = [6244,-178, 0.88, 1/4530, 85, -1.7, 0.0053]
	sol = least_squares(tot_sys, x0)
	print(sol.x)

	return sol

sol = optimize_params(X)
#solx = [ 6.24384105e+03, -1.77572038e+02,  4.71887563e+01,  4.68504592e-05, 8.50030131e+01, -1.94590999e+00,  4.52484986e-03]
#solx1 = [-3.88900025e+00,  1.62035460e+00, -5.13906478e-01, -2.27610625e+00, 2.29343983e+01,  3.89923137e-03,  3.70120867e-07]
#solx2 = [ 1.78060302, -0.27683487, -0.01693492,  0.01989211, -5.79270426,  1.58539399, -0.01642671]
out = np.zeros((6, np.shape(X)[1]))
for i in range(np.shape(X)[1]):
	out[:,i] = np.squeeze(sys_w_params(X[:,i], input_WJ_w_params, solx2))

plt.figure()
plt.subplot(311)
plt.plot(X[-1], out[3])
plt.plot(X[-1], X[4])
plt.legend(['model', 'data'])
plt.ylabel('du')
plt.grid()
plt.subplot(312)
plt.plot(X[-1], out[4])
plt.plot(X[-1], X[5])
plt.legend(['model', 'data'])
plt.ylabel('dv')
plt.grid()
plt.subplot(313)
plt.plot(X[-1], out[5])
plt.plot(X[-1], X[6])
plt.legend(['model', 'data'])
plt.ylabel('dr')
plt.grid()




#force_list = np.array(force_list)
# plt.figure()
# plt.plot(X[-1], force_list[:,0])
# plt.plot(X[-1], force_list[:,1])
# plt.plot(X[-1], force_list[:,2])
# plt.legend(['x', 'y', 'z'])



plt.show()
exit()


"""
solx0 = [ 6.24384105e+03, -1.77572038e+02,  4.71887563e+01,  4.68504592e-05, 8.50030131e+01, -1.94590999e+00,  4.52484986e-03]
solx1 = [-3.88900025e+00,  1.62035460e+00, -5.13906478e-01, -2.27610625e+00, 2.29343983e+01,  3.89923137e-03,  3.70120867e-07]
solx2 = [ 1.78060302, -0.27683487, -0.01693492,  0.01989211, -5.79270426,  1.58539399, -0.01642671]

"""








plt.figure()
plt.subplot(511)
plt.plot(X[-1], tauB_array[0,:])
plt.ylabel('fx')
plt.subplot(512)
plt.plot(X[-1], tauB_array[1,:])
plt.ylabel('fy')
plt.subplot(513)
plt.plot(X[-1], tauB_array[2,:])
plt.ylabel('nz')
plt.subplot(514)
plt.plot(X[-1], X[-3])
plt.ylabel('nozzle')
plt.subplot(515)
plt.plot(X[-1], X[-4])
plt.ylabel('RPM')


plt.figure()
plt.subplot(311)
plt.plot(X[-1], out[3])
plt.plot(X[-1], X[4])
plt.grid()
plt.legend(['Model', 'data'])

plt.subplot(312)
plt.plot(X[-1], out[4])
plt.plot(X[-1], X[6])
plt.grid()
plt.legend(['Model', 'data'])

plt.subplot(313)
plt.plot(X[-1], out[5])
plt.plot(X[-1], X[5])
plt.grid()
plt.legend(['Model', 'data'])





plt.show()