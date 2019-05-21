import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.integrate import cumtrapz

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

	thrust = rpm_scale * thrust_unscaled *0.5


	#waterjet port
	Fx = thrust*np.cos(delta_nozzle)
	Fy = thrust*np.sin(delta_nozzle)

	#moment
	Nz_port = (lever_CGtowj_port[0]*Fy)- (lever_CGtowj_port[1]*Fx)
	Nz_stb = (lever_CGtowj_stb[0]*Fy)- (lever_CGtowj_stb[1]*Fx)

	tau_b =  [2*Fx, 2*Fy, Nz_port + Nz_stb]

	return tau_b

def new_system_model(params, states, tau_b):
    tauB = np.zeros((3,1))
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
    m = 5000#params[0]
    Iz = params[1] #TODO Pz, sjekk at dette stemmer. Burde være Px?????

    # center of gravity
    xg = 0
    yg = 0

    # added mass
    Xdu = -params[2]
    Ydv = -params[3]
    Ydr = -params[4]
    Ndv = -params[5]
    Ndr = -params[6]

    # damping (borrowed from Loe 2008)
    Xu = -params[7]
    Yv = -params[8]
    Yr = -params[9]
    Nr = -params[10]

    Xuu = -params[11] #TODO gang med 1.8?? Hvorfor gjør Thomas det
    Yvv = -params[12]
    Nrr = -params[13] # TODO eller 0??

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

def ls_sys(X): 

    def fun(params):
        eta_dot_nu_dot = np.zeros((6, len(X[-1])))
        for i in range(np.shape(X)[1]):
            tau_b = jet_model(X[0:3, i], X[-4, i], X[-3, i]) # nu, jet_rpm, delta_nozzle

            states = np.append(X[6, i], X[0:3, i])

            eta_dot_nu_dot[:, i] = np.squeeze(new_system_model(params,states, tau_b))


        #weight the error
        W = [1, 1, 10]

        sq_err = np.add(np.add(((X[3] - eta_dot_nu_dot[3])**2)*W[0],((X[4] - eta_dot_nu_dot[4])**2)*W[1]), ((X[5] - eta_dot_nu_dot[5])**2)*W[2])
        mse = np.sum(sq_err)/len(sq_err)
        print('Mean Squared Error:', mse)
        print('Params: ', params)
        return sq_err



    #x0 = [6000,21000,0,0,0,0,0,50,200,0,1281,200,2000,20000]
    x0 =[5.35567816e+04, 7.00303675e+05, 4.69368829e+01, 1.01931936e+04, 1.30953182e-25, 3.11550449e+05, 6.78559121e+05, 4.21478305e-06, 2.56368714e+04, 3.38745246e-15, 3.95788347e-29, 2.64016266e-06, 1.66729827e+04, 1.66440595e+05] 
    sol = least_squares(fun, x0, bounds=(0,np.inf), ftol=0.001)
    print(sol.x)
    return sol.x



### ---load data---
path = '/home/gislehalv/Master/scripts/standard_model/'

#X = load_data(path)
X = np.load(path +'Data_cut01.npy')

lpp = 10.5
g = 9.81

U = np.sqrt(np.add(X[0, :]**2, X[1, :]**2))
Froude = U/np.sqrt(g*lpp)


# plt.figure()
# plt.plot(X[-1], Froude)
# plt.plot(X[-1], np.ones(len(X[-1]))*0.4)
# plt.plot(X[-1], np.ones(len(X[-1])))


## remove U > 7m/s
remove = True
if remove:

    index = []
    for i in range(np.shape(X)[1]):
        if Froude[i] > 0.4:
            index.append(i)

    X = np.delete(X, index, axis = 1)



index = list(range(0, 10000))+ list(range(30000,len(X[1])))
index_sub = []
for i in index:
    if np.random.rand() > 0.8:
        index_sub.append(i)

XX = X[:,index_sub]
### ----divide in subsets

# - rand selection - not the best considering bias towards overfitting
# index_tot = list(range(np.shape(X)[1]))
# index = np.random.choice(index_tot, size = int(len(index_tot)/4))
# XX = X[:, index]


# - The first forth-part of the data
#XX = X[:, 0: int(np.shape(X)[1]/4)]


# - first fourth and last fourth
#XX = np.concatenate((X[:, 0:int(np.shape(X)[1]/4)], X[:, int(np.shape(X)[1] *3/4):]), axis = 1)

# - first fourth and last eights
#XX = np.concatenate((X[:, 0:int(np.shape(X)[1]/8)], X[:, int(np.shape(X)[1] *4/8):]), axis = 1)



#- all the data
#XX = X.copy()

#opt_par = ls_sys(XX)


###sols
#opt_par = [9.45653632e+03, 8.74004580e+04, 6.29805992e+00, 7.97343523e+03, 7.44359556e-26, 1.17006958e-23, 5.21583270e+04, 9.45448584e+02, 6.09398994e+03, 4.95830970e+03, 7.57110011e-09, 4.11559744e-15, 1.20542232e+01, 7.50726738e+03]
#opt_par = [9.48949130e+03, 8.59028923e+04, 2.03238896e+02, 6.78129627e+03, 1.15084553e-16, 1.43375341e-18, 5.06607614e+04, 9.63721610e+02, 5.49165290e+03, 4.99143494e+03, 2.65407221e-01, 3.85321293e-05, 1.84967522e+02, 1.47707008e+04]

##best solution yet, Mean Squared Error: 0.04381828100032658 - fitted with the first 1/4 th of the data
#opt_par = [8.98170680e+03, 2.13320352e+05, 2.01194553e-17, 1.10072779e+04, 2.06468226e-29, 2.29573058e-25, 1.92209340e+05, 2.75763260e+02, 8.07217059e+03, 1.71623610e+04, 9.58673612e-2,8 1.90514441e+02, 9.54182435e-28, 5.05916109e+03]

### with m = 5000, mse = 0.04541673188233341:
#opt_par = [5.00000000e+03, 4.49852483e+05, 3.89974427e+03, 1.74880902e+04, 4.99271690e+03, 2.50219179e-25, 4.46469448e+05, 2.44977603e+02, 7.05395914e+03, 3.95436722e+04, 8.37771906e-27, 1.95822797e+02, 2.50555167e+03, 3.42500064e+04]

### m = 6000 - looks good for u and v but not r at higher velocities - used 1/8 first and the last half to fit
#opt_par = [6.00000000e+03, 3.48432893e+04, 4.28538418e+03, 1.49443831e+03, 2.98471867e-24, 2.00906022e-02, 1.38430814e+04, 1.85783049e+03, 1.89370166e+03, 7.60152214e+03, 3.79391200e-08, 3.22223105e+00, 1.37189312e+02, 9.05423308e+03]
#opt_par = [5.23340622e+04, 8.82754037e+04, 5.18420311e+03, 3.03920329e+03, 6.05154260e-18, 4.24636553e+04, 7.60825580e+04, 2.74993563e+01, 1.62018706e+04, 1.22892868e-13, 7.22177584e-03, 3.31242136e+01, 1.77205621e+04, 2.48273186e+03]

### m = 5000 - final model
opt_par = [5.000e+04, 1.57472121e+06, 4.46839484e+04, 7.18370633e+04, 2.05674443e-01, 1.28871321e+06, 2.67905789e+06, 2.39711115e+02, 1.31828582e+04, 2.37723162e+05, 1.94302007e+02, 8.99313430e+00, 3.45319720e+04, 1.58336114e+05]



### ---Simulate---
tau_b = np.zeros((3, len(X[-1])))
eta_dot_nu_dot = np.zeros((6, len(X[-1])))

for i in range(len(X[-1])):
	tau_b[:, i] = jet_model(X[0:3, i], X[-4, i], X[-3, i])

	states = np.append(X[6, i], X[0:3, i])

	eta_dot_nu_dot[:, i] = np.squeeze(new_system_model(opt_par,states, tau_b[:, i]))



##---- cumTrapz
u = cumtrapz(eta_dot_nu_dot[3], dx = 0.05, initial = eta_dot_nu_dot[3,0])
v = cumtrapz(eta_dot_nu_dot[4], dx = 0.05, initial = eta_dot_nu_dot[4,0])
r = cumtrapz(eta_dot_nu_dot[5], dx = 0.05, initial = eta_dot_nu_dot[5,0])

u_hat = cumtrapz(X[3], dx = 0.05, initial = X[0,0])

### ---- Plot ---
#with plt.xkcd():


## inputs
# plt.figure()
# plt.subplot(511)
# plt.title('Force and Jet Inputs')
# plt.plot(X[-1], tau_b[0, :])
# plt.ylabel('Fx [N]')
# plt.grid()

# plt.subplot(512)
# plt.plot(X[-1], tau_b[1, :])
# plt.ylabel('Fy [N]')
# plt.grid()

# plt.subplot(513)
# plt.plot(X[-1], tau_b[2, :])
# plt.ylabel('Nz [Nm]')
# plt.grid()

# plt.subplot(514)
# plt.plot(X[-1], X[-4])
# plt.ylabel('Jet RPM [rpm]')
# plt.grid()

# plt.subplot(515)
# plt.plot(X[-1], X[-3])
# plt.ylabel('Nozzle Angle [rad]')
# plt.grid()



# acc
plt.figure()
plt.subplot(311)
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), X[3])
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), eta_dot_nu_dot[3])
plt.ylabel('$\dot{u}$')
plt.legend(['data', 'model'])
plt.grid()

plt.subplot(312)
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), X[4])
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), eta_dot_nu_dot[4]) 
plt.ylabel('$\dot{v}$')
plt.legend(['data', 'model'])	
plt.grid()

plt.subplot(313)
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), X[5])
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), eta_dot_nu_dot[5]) 
plt.ylabel('$\dot{r}$')
plt.xlabel('Time [s]')
plt.legend(['data', 'model'])
plt.grid()


plt.figure()
plt.subplot(311)
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), X[0])
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), u)
plt.ylabel('u')
plt.legend(['data', 'model'])
plt.grid()

plt.subplot(312)
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), X[1])
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), v) 
plt.ylabel('v')
plt.legend(['data', 'model'])   
plt.grid()

plt.subplot(313)
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), X[2])
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), r) 
plt.ylabel('r')
plt.xlabel('Time [s]')
plt.legend(['data', 'model'])
plt.grid()




plt.show()