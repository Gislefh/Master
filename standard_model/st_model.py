import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

### --- jet model ---
def jet_model(nu, jet_rpm, delta_nozzle):
	#constants
	lever_CGtowj_port = [-3.82, -0.475]
	lever_CGtowj_stb = [-3.82, 0.475]

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





def ls_wj(X, sys):

	#params = [a0, a1, a2, r0, r1, r2]
	def new_wj_model(params, nu, jet_rpm, delta_nozzle):
		#constants
		lever_CGtowj_port = [-3.82, -0.475]
		lever_CGtowj_stb = [-3.82, 0.475]

		#rpm2thrust
		speed = nu[0]		 ###* 1.94384 # knots no need
		a0 = params[0]
		a1 = params[1]
		a2 = params[2]
		thrust_unscaled = a0 + a1*speed + a2*(speed**2)

		r0 = params[3]
		r1 = params[4]
		r2 = params[5]

		rpm_scale = r0 + r1*jet_rpm + r2 * (jet_rpm **2) # no need to scale

		thrust = rpm_scale * thrust_unscaled


		#waterjet port
		Fx = thrust*np.cos(delta_nozzle)
		Fy = thrust*np.sin(delta_nozzle)

		#moment
		Nz_port = (lever_CGtowj_port[0]*Fy)- (lever_CGtowj_port[1]*Fx)
		Nz_stb = (lever_CGtowj_stb[0]*Fy)- (lever_CGtowj_stb[1]*Fx)

		tau_b =  [2*Fx, 2*Fy, Nz_port + Nz_stb]

		return tau_b

	def fun(params):
		eta_dot_nu_dot = np.zeros((6, len(X[-1])))
		for i in range(np.shape(X)[1]):
			tau_b = new_wj_model(params, X[0:3, i], X[-4, i], X[-3, i]) # nu, jet_rpm, delta_nozzle

			states = np.append(X[6, i], X[0:3, i])

			eta_dot_nu_dot[:, i] = np.squeeze(sys(states, tau_b))

		#du = eta_dot_nu_dot[3]
		#dv = eta_dot_nu_dot[4]
		#dr = eta_dot_nu_dot[5]

		#optimize for du
		all_st = True
		just_du = False

		if just_du:
			sq_err = ((X[3] - eta_dot_nu_dot[3])**2)
			mse = np.sum(sq_err)/len(sq_err)
			print('Mean Squared Error:', mse)
			return sq_err

		if all_st:
			sq_err = np.add(np.add(((X[3] - eta_dot_nu_dot[3])**2),((X[4] - eta_dot_nu_dot[4])**2)), ((X[5] - eta_dot_nu_dot[5])**2))
			mse = np.sum(sq_err)/len(sq_err)
			print('Mean Squared Error:', mse)
			return sq_err



	x0 = [1000,-100,1,10,-1,0.01]
	sol = least_squares(fun, x0)
	print(sol.x)




### ---load data---
path = '/home/gislehalv/Master/scripts/standard_model/'

#X = load_data(path)
X = np.load(path +'Data_cut01.npy')

#X = np.delete(X, list(range(1000, np.shape(X)[1])), axis = 1)

ls_wj(X, sys)


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
plt.plot(X[-1], eta_dot_nu_dot[4] *(-1/2)) ## minus passer bedre? hmmm, passer godt med *(-1/2)
plt.ylabel('dv')
plt.legend(['data', 'model'])	
plt.grid()

plt.subplot(313)
plt.plot(X[-1], X[5])
plt.plot(X[-1], eta_dot_nu_dot[5] *(-1/10)) ## minus passer bedre? hmmm, passer godt med *(-1/10)
plt.ylabel('dr')
plt.legend(['data', 'model'])
plt.grid()




plt.show()