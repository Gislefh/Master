

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import rosbag
from scipy import signal


### --- jet model ---
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
	Fx = thrust*np.cos(delta_nozzle)
	Fy = thrust*np.sin(delta_nozzle)

	#moment
	Nz_port = (lever_CGtowj_port[0]*Fy)- (lever_CGtowj_port[1]*Fx)
	Nz_stb = (lever_CGtowj_stb[0]*Fy)- (lever_CGtowj_stb[1]*Fx)

	tau_b =  [2*Fx, 2*Fy, Nz_port + Nz_stb]

	return tau_b

#states = yaw, u, v, r 
def sys(states, tau_b):
	

	tauB = np.zeros((3,1))
	#tauB = input(X)
	tauB[:, 0] = tau_b

	nu = np.zeros((3,1))
	nu[0,0] = states[1]
	nu[1,0] = states[2]
	nu[2,0] = states[3]

	yaw = states[0];

	u = nu[0]
	v = nu[1]   
	r = nu[2]


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
	Dnl_g = - np.array([[np.squeeze(Xuu*np.abs(u)), 0, 0],
					[0, np.squeeze(Yvv*abs(v)), 0],
					[0, 0, np.squeeze(Nrr*abs(r))]])		

	

	D_g = np.add(Dl_g, Dnl_g)
	
	

	eta_dot = np.dot(J,nu);
	nu_dot = np.dot(np.linalg.inv(M), (tauB - np.dot(C_g, nu) - np.dot(D_g, nu)))

	out = np.concatenate((eta_dot, nu_dot))
	

	return out

#X = [u_smooth, v_smooth, r_smooth, du_smo, dv_smo, dr_smo, yaw, jet_rpm, nozzle_angle, bucket, interp_arr]
def open_bag_w_yaw(path, plot = False, thr_bucket = False, filter_cutoff = 0.025, return_raw_data = False):
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
		nozzle_angle =	np.interp(interp_arr, jet_data[2, :], jet_data[1, :]) #  
		bucket = 		np.interp(interp_arr, jet_data[2, :], jet_data[3, :]) # from [-100, to 100]
		yaw = 			np.interp(interp_arr, nav_data[6, :], nav_data[2, :]) # yaw
		return jet_rpm, nozzle_angle, bucket, yaw

	jet_rpm, nozzle_angle, bucket, yaw = interpolate(jet_data, nav_data, interp_arr)

	#preeprossesed matrix of variables.
	X = [u_smooth, v_smooth, r_smooth, du_smo, dv_smo, dr_smo, yaw, jet_rpm, nozzle_angle, bucket, interp_arr]
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

def load_data(path):
	# get data
	bag_1 = 'hal_control_2018-12-11-10-53-26_0' # low to medium speed zigzag + speed steps
	bag_2 = 'hal_control_2018-12-11-11-49-22_0' # high speed zigzag + 
	bag_3 = 'hal_control_2018-12-11-12-13-58_0' # speed steps
	bag_4 = 'hal_control_2018-12-11-12-13-58_0' # speed steps

	bagFile1 = path + bag_1 + '.bag'
	bagFile2 = path + bag_2 + '.bag'
	bagFile3 = path + bag_3 + '.bag'
	bagFile4 = path + bag_4 + '.bag'


	X1 = open_bag_w_yaw(bagFile1, plot=False, filter_cutoff = 0.1)
	X2 = open_bag_w_yaw(bagFile2, plot=False, filter_cutoff = 0.1)
	X3 = open_bag_w_yaw(bagFile3, plot=False, filter_cutoff = 0.1)
	X4 = open_bag_w_yaw(bagFile4, plot=False, filter_cutoff = 0.1)

	X = np.concatenate((X1,X2,X3,X4),axis = 1)

	X[6,:] = np.multiply(X[6,:], np.pi/180) # yaw deg2rad
	X[-3,:] = np.multiply(X[-3,:], np.pi/180) #nozzle deg2rad

	X[-1] = np.arange(0, len(X[-1])*0.05, 0.05)


	np.save('/home/gislehalv/Master/scripts/standard_model/Data_cut01', X)
	X = np.load('/home/gislehalv/Master/scripts/standard_model/Data_cut01.npy')
	return X


#X = np.load('/home/gislehalv/Master/scripts/standard_model/Data_cut01.npy')


### ---load data---
path = '/home/gislehalv/Master/scripts/standard_model/'

#X = load_data(path)
X = np.load(path +'Data_cut01.npy')

### ---Simulate---
tau_b = np.zeros((3, len(X[-1])))
eta_dot_nu_dot = np.zeros((6, len(X[-1])))

for i in range(len(X[-1])):
	tau_b[:, i] = jet_model(X[0:3, i], X[-4, i], X[-3, i])

	states = np.append(X[6, i], X[0:3, i])

	eta_dot_nu_dot[:, i] = np.squeeze(sys(states, tau_b[:, i]))



### ---- Plot ---
#with plt.xkcd():


## inputs
plt.figure()
plt.subplot(511)
plt.title('Force and Jet Inputs')
plt.plot(X[-1], tau_b[0, :])
plt.ylabel('Fx [N]')
plt.grid()

plt.subplot(512)
plt.plot(X[-1], tau_b[1, :])
plt.ylabel('Fy [N]')
plt.grid()

plt.subplot(513)
plt.plot(X[-1], tau_b[2, :])
plt.ylabel('Nz [Nm]')
plt.grid()

plt.subplot(514)
plt.plot(X[-1], X[-4])
plt.ylabel('Jet RPM [rpm]')
plt.grid()

plt.subplot(515)
plt.plot(X[-1], X[-3])
plt.ylabel('Nozzle Angle [rad]')
plt.grid()



# derivatives
plt.figure()
plt.subplot(311)
plt.plot(X[-1], X[3])
plt.plot(X[-1], eta_dot_nu_dot[3])
plt.ylabel('du')
plt.legend(['data', 'model'])
plt.grid()

plt.subplot(312)
plt.plot(X[-1], X[4])
plt.plot(X[-1], eta_dot_nu_dot[4]) ## minus passer bedre? hmmm, passer godt med *(-1/2)
plt.ylabel('dv')
plt.legend(['data', 'model'])	
plt.grid()

plt.subplot(313)
plt.plot(X[-1], X[5])
plt.plot(X[-1], eta_dot_nu_dot[5]) ## minus passer bedre? hmmm, passer godt med *(-1/10)
plt.ylabel('dr')
plt.legend(['data', 'model'])
plt.grid()




plt.show()