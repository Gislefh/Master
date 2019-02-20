import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp




def boat_simulation(time = [0, 30, 0.1], init_cond = [0, 0, 0, 0, 0, 0]):

	## System ###
	def sys(t,X):

		#tauB = np.zeros((3,1))
		tauB = inp


		eta = np.zeros((3,1))
		eta[0,0] = X[0]
		eta[1,0] = X[1]
		eta[2,0] = X[2]

		nu = np.zeros((3,1))
		nu[0,0] = X[3]
		nu[1,0] = X[4]
		nu[2,0] = X[5]

		x = eta[0];
		y = eta[1];
		yaw = eta[2];


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

		C_g = np.multiply(C_RB_g, C_A_g)

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
	

	inp = np.zeros((3,1))
	inp[0,0] = 10000
	#inp[1,0] = -2

	sol = solve_ivp(sys, [t[0], t[-1]], init_cond, vectorized = True, max_step = t_step, t_eval = t)


	sol_dot = np.zeros((np.shape(sol.y)[0], np.shape(sol.y)[1]))



	for tmp, i in enumerate(t):
		
		sol_dot[:,tmp] = np.squeeze(sys(i, sol.y[:,tmp]))


	plt.figure()
	plt.plot(sol.y[0], sol.y[1])
	plt.figure()
	plt.plot(t,sol_dot[0,:])
	plt.figure()
	plt.plot(t,sol_dot[3,:])
	plt.show()
	

	return None

boat_simulation()

"""
#from Matlab

% position and orientation
x = eta(1);
y = eta(2);
yaw = eta(3);

% velocities
u = nu(1);
v = nu(2);
r = nu(3);

% mass and moment of inertia
m = 4935.14;
Iz = 20928; %TODO Pz, sjekk at dette stemmer. Burde være Px?????

% center of gravity
xg = 0;
yg = 0;

% added mass
Xdu = 0;
Ydv = 0;
Ydr = 0;
Ndv = 0;
Ndr = 0;

% damping (borrowed from Loe 2008)
Xu = -50;
Yv = -200;
Yr = 0;
Nr = -1281;

Xuu = -135*1.8; %TODO gang med 1.8?? Hvorfor gjør Thomas det
Yvv = -2000;
T = 4;
K = 0.5;
Nrr = -Iz*1.1374*K/T; % TODO eller 0??

% transformation matrix, equal to rotation matrix about z-axis
J = [cos(yaw) -sin(yaw) 0;
     sin(yaw) cos(yaw)  0;
     0        0         1];

% rigid body mass
M_RB = [m 0 0;
        0 m 0;
        0 0 Iz];

% hydrodynamic added body mass
M_A = -[Xdu 0 0;
        0 Ydv Ydr;
        0 Ndv Ndr];
 
% total mass
M = M_RB + M_A;

%Coriolis
C_RB_g = zeros(3,3);
C_RB_g(1,3) = -m*(xg*r+v);
C_RB_g(3,1) = -C_RB_g(1,3);
C_RB_g(2,3) = m*u;
C_RB_g(3,2) = -C_RB_g(2,3);

C_A_g = zeros(3,3);
C_A_g(1,3) = Ydv*v+Ydr*r;
C_A_g(3,1) = -C_A_g(1,3);
C_A_g(2,3) = Xdu*u;
C_A_g(3,2) = -C_A_g(2,3);

C_g = C_RB_g + C_A_g;

%Linear damping
Dl_g = -diag([Xu, Yv, Nr]);
Dl_g(2,3) = -Yr;
Dl_g(3,2) = -Nr;

%Nonlinear damping
Dnl_g = -diag([Xuu*abs(u),Yvv*abs(v), Nrr*abs(r)]);

D_g = Dl_g+Dnl_g;

eta_dot = J*nu;
nu_dot = inv(M)*(tauB - C_g*nu - D_g*nu);
"""