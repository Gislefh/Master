import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.integrate import cumtrapz
from sympy import sympify, cos, sin, expand, collect, Lambda, lambdify, symbols

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


##takes in string -> gives out the function for the string
##takes in string -> gives out the function for the string
def str2func(exp_str):
    u, v, r, delta_t, delta_n = symbols('u v r delta_t delta_n')
    f = lambdify((u,v,r, delta_t, delta_n), exp_str, 'numpy')
    return f


def str2func_force(exp_str):
    u, v, r, fx, fy, fz = symbols('u v r fx, fy, fz')
    f = lambdify((u,v,r, fx, fy, fz), exp_str, 'numpy')
    return f


### ---load data---
path = '/home/gislehalv/Master/scripts/standard_model/'

#X = load_data(path)
X = np.load(path +'Data_cut01.npy')

#test
#X = X[:, list(range(75000, len(X[0])))]

#val 
X = np.concatenate((X[:,11511:18500], X[:,27912:32693], X[:,56531:60714], X[:,70475:74193]),axis = 1)

#index = list(range(11511,18500)) + list(range(27912,32693)) +list(range(56531,60714)) + list(range(70475,74193)) + list(range(75000, np.shape(X)[1]))

#train
#X = np.delete(X, index, 1)


lpp = 10.5
g = 9.81

U = np.sqrt(np.add(X[0, :]**2, X[1, :]**2))
Froude = U/np.sqrt(g*lpp)


##phases
remove = True
if remove:
    index = []
    for i in range(np.shape(X)[1]):
        if Froude[i] > 0.4:
            index.append(i)
    X = np.delete(X, index, axis = 1)

#remove data with bucket < 95
index = []
for i in range(np.shape(X)[1]):
    if np.shape(X)[1] > i:
        if X[-2, i] < 95:
            index.append(i)
X = np.delete(X, index, 1)




### m = 5000 - final ST model params - diss phase
opt_par = [5.000e+04, 1.57472121e+06, 4.46839484e+04, 7.18370633e+04, 2.05674443e-01, 1.28871321e+06, 2.67905789e+06, 2.39711115e+02, 1.31828582e+04, 2.37723162e+05, 1.94302007e+02, 8.99313430e+00, 3.45319720e+04, 1.58336114e+05]



### ---Simulate ST model
tau_b = np.zeros((3, len(X[-1])))
eta_dot_nu_dot = np.zeros((6, len(X[-1])))

for i in range(len(X[-1])):
	tau_b[:, i] = jet_model(X[0:3, i], X[-4, i], X[-3, i])

	states = np.append(X[6, i], X[0:3, i])

	eta_dot_nu_dot[:, i] = np.squeeze(new_system_model(opt_par,states, tau_b[:, i]))


#### --  new model with actuator dynamics ---- 


#deg inputs
#X[8] = X[8] * (180/np.pi)
#du_ac_dyn_str = '-5.015465314312387*r**2 + 0.010787502174784275*u**2 + 0.007307473954963339*sin(u)**2 + 3.135414581806233e-07*2*delta_t**2 -0.00021039428976226543*delta_t*v + 7.828050576367141e-05*delta_t*cos(u) + 0.884678750533654*r*v + 1.0541754076381125*r*cos(r) + 0.06584852953272069*r*cos(u) + 0.019297315056831632*u*v -0.26897900755277804*u*cos(r) + 0.01902230024683776*u*cos(u) + 0.2511746970208981*v*cos(r) -0.06029356401213615*v*sin(u) -0.37978015576637336*cos(r)*cos(u) + 0.31897338340622394*sin(u)*cos(r) + 0.0379281092537167*sin(u)*cos(u) + 4.966699849507045e-05*2*delta_t*cos(r) + 0.056841370146047066*2*r*u -0.13838713020550086*2*r*sin(u) -0.010981246609847131*2*u*sin(u) -0.0003283640961381819*3*delta_t*r -9.697605228418569e-06*3*delta_t*u -3.226831355353753e-05*3*delta_t*sin(u)'
#dv_ac_dyn_str = '-3.952212504979791e-06*delta_n**3*u -4.323806251907643e-06*delta_n**3*v + 3.633763321792768e-06*delta_n**2*u**2 -1.831870024171225e-07*delta_n*delta_t*u**2 + 0.0003823435101753467*delta_n*u**2*sin(cos(r)) -2.1912674505216164e-09*delta_n**2*delta_t*u -8.612870897527819e-07*delta_n**2*delta_t*v -3.3142056104988395e-09*delta_t*u**2*sin(cos(r)) + 7.719118258445558e-05*delta_n**2*u*v + 5.402037395386755e-06*delta_n**2*u*sin(cos(r)) + 0.0005809985939579979*delta_n**2*v*sin(cos(r)) + 6.585244046079131e-07*delta_n*delta_t*u*v -7.821381053574049e-07*delta_n*delta_t*u*sin(cos(r)) + 6.313586874164905e-06*delta_n*delta_t*v*sin(cos(r)) -0.0023077638335907077*delta_n*u*v*sin(cos(r)) -1.4485249323632313e-05*delta_t*u*v*sin(cos(r))'
#dr_ac_dyn_str = '-1.8790570516231525e-06*delta_n*delta_t + 0.0011251273020889668*delta_n*v + 4.9114361114335354e-09*delta_n*delta_t**2 + 0.03279426174278255*delta_n*r**2 -5.946117028410193e-08*delta_t*u**2 + 1.1306310777796303e-05*delta_t*v**2 + 3.3190501269843805e-10*delta_t**2*u -0.0011467582186509059*u*v**2 + 9.018856112370991e-08*delta_t**2*v + 0.0012651422425927622*u**2*v + 0.015191554927764361*r**2*u**2 + 1.1375006351434954e-06*delta_n*delta_t*r -1.9872024426422902e-07*delta_n*delta_t*u -1.7916605582820277e-06*delta_n*delta_t*v -9.185627609026802e-05*delta_n*delta_t*r**2 -1.6644052762981315e-09*delta_n*delta_t**2*r -4.452349825126672e-05*delta_n*u*v + 0.0169578792393221*delta_n*r**2*u + 4.907055384612531e-06*delta_t*r*u**2 + 5.717806295840299e-05*delta_t*r**2*u -0.000813914658875696*delta_t*r**2*v -3.9646083548925386e-08*delta_t**2*r*u + 4.475393143190829e-09*delta_t**2*r*v + 0.12220372961142889*r**2*u*v -9.051680317981846e-06*2*delta_t*u*v -1.438935345690965e-06*delta_n*delta_t*r*u + 1.522701530279913e-05*delta_t*r*u*v'

#rad inputs
du_ac_dyn_str = ' -6.401723315032696e-07*delta_t**2*Abs(r) -1.7510597059935594e-07*delta_t**2*Abs(v) + 9.536754448195308e-14*delta_t**3*Abs(delta_t) -0.00028974428495586076*delta_t*u*Abs(r) + 5.8123681708149334e-05*delta_t*u*Abs(v) + 1.6226878593276234e-08*delta_t*u**2*Abs(delta_t) + 1.6751406875743834e-07*delta_t**2*u*Abs(r) -1.772521622200421e-08*delta_t**2*u*Abs(v) + 1.8989290167065055e-14*delta_t**3*u*Abs(delta_t) -7.242081617042239e-12*delta_t**2*u**2*Abs(delta_t) -1.1926182408954693e-11*2*delta_t**2*u*Abs(delta_t) -0.1550877197885252*u -0.00013731176156493025*delta_t*u + 0.0008556537700173266*delta_t'
#dv_ac_dyn_str = '-0.8197423784238715*delta_n*r + 0.0005133412411933991*delta_t*r + 2.1859938852579498e-07*delta_t*u**2 -0.20030157909449287*r*u -0.3261752406421472*r*v -0.0410621185967085*r*u**2 -1.879603411062817*r*Abs(v) -2.8131640937430316e-10*delta_t**2*u -9.770874306896829e-07*delta_t**2*sin(r) -9.552637737276549e-05*delta_n*delta_t*u + 0.0006802415512793536*delta_n*delta_t*sin(r) + 0.37215256254539186*delta_n*r*u -2.1981078379108823*delta_n*r*sin(r) -0.011262602130010783*delta_t*r*u -0.0030855400943216454*delta_t*r*sin(r) -2.376676901135287e-05*delta_t*u*v -2.4856342287284556e-06*delta_t*u*Abs(v) + 0.011720234366986961*delta_t*u*sin(r) -0.0005642286233415485*delta_t*v*sin(r) + 0.002153497891142664*delta_t*sin(r)*Abs(v) + 0.1599792930227779*r*u*v -0.22616087998606016*r*u*Abs(v) + 0.3690980043692882*r*u*sin(r) -5.0327802641935815*r*v*sin(r) -0.6830893859825693*r*sin(r)*Abs(v)'
dv_ac_dyn_str = '-0.12015367507328875*delta_n*u -0.04246619170961782*delta_n*v + 2.695536193145574e-06*delta_t*u -0.0002946719745563252*delta_t*v -0.2086702090256655*r*u -0.006945684834470195*r*v -0.002948500848703293*u*Abs(v) + 0.01884659221662005*v*Abs(v)'
dr_ac_dyn_str = '0.005122559403305953*delta_n**3 + 0.005606333600127404*v**3 + 1.0052428722221043e-07*delta_n*delta_t**2 + 0.0016046518339934424*delta_n**2*u -0.0014994607098292728*u*v**2 + 7.50367067278903e-08*delta_t**2*v -3.438906142914531e-05*2*delta_n**2*delta_t + 4.645482279500516e-06*2*delta_t*v**2 + 0.03719878448922598*3*delta_n*v**2 + 0.04632037045510053*3*delta_n**2*v + 3.121939802249246e-06*delta_n*delta_t*u -6.894204325890214e-06*delta_t*u*v + 0.003485218313295839*2*delta_n*u*v -9.361928791588013e-06*4*delta_n*delta_t*v'


du_ac_dyn_func = str2func(du_ac_dyn_str)
dv_ac_dyn_func = str2func(dv_ac_dyn_str)
dr_ac_dyn_func = str2func(dr_ac_dyn_str)

du_ac_dyn_array = du_ac_dyn_func(X[0], X[1], X[2], X[7], X[8])
dv_ac_dyn_array = dv_ac_dyn_func(X[0], X[1], X[2], X[7], X[8])
dr_ac_dyn_array = dr_ac_dyn_func(X[0], X[1], X[2], X[7], X[8])



### -- new model force --
file = '/home/gislehalv/Master/Data/numpy_data_from_bag_force/all_bags_cut1.npy'
X_force = np.load(file)

#test
#X_force = X_force[:, list(range(75000, len(X_force[0])))]

#val 
X_force= np.concatenate((X_force[:,11511:18500], X_force[:,27912:32693], X_force[:,56531:60714], X_force[:,70475:74193]),axis = 1)
#index = list(range(11511,18500)) + list(range(27912,32693)) +list(range(56531,60714)) + list(range(70475,74193)) + list(range(75000, np.shape(X)[1]))

#train
#X_force = np.delete(X_force, index, 1)


U_force = np.sqrt(np.add(X_force[0, :]**2, X_force[1, :]**2))
Froude_force = U_force/np.sqrt(g*lpp)


if remove:
    index = []
    for i in range(np.shape(X_force)[1]):
        if Froude_force[i] > 0.4:
            index.append(i)
    X_force = np.delete(X_force, index, axis = 1)

#remove data with bucket < 95
index = []
for i in range(np.shape(X_force)[1]):
    if np.shape(X_force)[1] > i:
        if X_force[-2, i] < 95:
            index.append(i)
X_force = np.delete(X_force, index, 1)


du_force_str = '0.000264871203432643*fx -5.104448463432132e-07*fx*u**3 -2.773742056836302e-06*Abs(fz*v) + 2.649152675182239e-09*fx*Abs(fx) -0.00010529881090182071*r*v*Abs(fx) -0.037022654378220055*u**2 + 0.0020864289417205986*u**3 -1.722147024416803e-05*fx*r -9.437055733162733e-05*fx*u + 1.348464006791731e-05*fx*u**2 + 0.04778449754855396*r*u + 2.2407149473159294*r*v + 0.00947558535982701*u*v -0.0007198331382015782*u**2*v'
dv_force_str = '0.00017726146909316173*u**2 -2.9826827341053195e-07*fx*u -5.111840115947252e-05*fx*v + 4.192962483021315e-06*fz*u + 2.1305919250285774e-07*fz*v -0.20797031166711719*r*u + 0.27756501493808916*r*v -0.008344792948138165*u*v + 5.11254234557995e-06*u*Abs(fy) -1.0616675717356819e-05*v*Abs(fy)'
dr_force_str = '1.0842307778833146e-05*fx*v -5.1900734221065225e-09*fx*u**2 -5.619128822111664e-06*fy*v -2.1858712064244324e-08*fy*u**2 -4.099291394108764e-09*fx**2*r + 6.760385933163654e-12*fx**2*u -1.4636965662517697e-09*fx*fy*r + 3.040026213995778e-10*fx*fy*u + 1.9071315513956064e-06*fx*r*u -3.853593934102024e-06*fy*r*u + 2.5257972496607324e-05*fy'

du_force_func = str2func_force(du_force_str)
dv_force_func = str2func_force(dv_force_str)
dr_force_func = str2func_force(dr_force_str)

du_force_array = du_force_func(X_force[0], X_force[1], X_force[2], X_force[6], X_force[7], X_force[8])
dv_force_array = dv_force_func(X_force[0], X_force[1], X_force[2], X_force[6], X_force[7], X_force[8])
dr_force_array = dr_force_func(X_force[0], X_force[1], X_force[2], X_force[6], X_force[7], X_force[8])




##---- cumTrapz
u_st = cumtrapz(eta_dot_nu_dot[3], dx = 0.05, initial = eta_dot_nu_dot[3,0])
v_st = cumtrapz(eta_dot_nu_dot[4], dx = 0.05, initial = eta_dot_nu_dot[4,0])
r_st = cumtrapz(eta_dot_nu_dot[5], dx = 0.05, initial = eta_dot_nu_dot[5,0])

u_f = cumtrapz(du_force_array, dx = 0.05, initial = du_force_array[0])
v_f = cumtrapz(dv_force_array, dx = 0.05, initial = dv_force_array[0])
r_f = cumtrapz(dr_force_array, dx = 0.05, initial = dr_force_array[0])

u_ac = cumtrapz(du_ac_dyn_array, dx = 0.05, initial = du_ac_dyn_array[0])
v_ac = cumtrapz(dv_ac_dyn_array, dx = 0.05, initial = dv_ac_dyn_array[0])
r_ac = cumtrapz(dr_ac_dyn_array, dx = 0.05, initial = dr_ac_dyn_array[0])



### ---- Plot ---


# acc
plt.figure()
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), X[3], linewidth = 0.5)
#plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), eta_dot_nu_dot[3])
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), du_ac_dyn_array)
plt.plot(np.arange(0, len(X_force[-1])*0.05, 0.05), du_force_array)
plt.ylabel('$\dot{u}$')
plt.legend(['Data', 'Model with Actuator Dynamics', 'Model with Force as Input'])
plt.grid()


plt.figure()
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), X[4], linewidth = 0.5)
#plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), eta_dot_nu_dot[4]) 
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), dv_ac_dyn_array)
plt.plot(np.arange(0, len(X_force[-1])*0.05, 0.05), dv_force_array)
plt.ylabel('$\dot{v}$')
plt.legend(['Data' ,'Model with Actuator Dynamics', 'Model with Force as Input'])
plt.grid()

plt.figure()
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), X[5], linewidth = 0.5)
#plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), eta_dot_nu_dot[5]) 
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), dr_ac_dyn_array)
plt.plot(np.arange(0, len(X_force[-1])*0.05, 0.05), dr_force_array)
plt.ylabel('$\dot{r}$')
plt.xlabel('Time [s]')
plt.legend(['Data',  'Model with Actuator Dynamics', 'Model with Force as Input'])
plt.grid()


#velo
plt.figure()
plt.subplot(311)
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), X[0])
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), u_st)
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), u_ac)
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), u_f)
plt.ylabel('u')
plt.legend(['Data', 'Standard model', 'Model with Actuator Dynamics', 'Model with Force as Input'])
plt.grid()

plt.subplot(312)
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), X[1])
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), v_st)
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), v_ac)
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), v_f) 
plt.ylabel('v')
plt.legend(['Data', 'Standard model', 'Model with Actuator Dynamics', 'Model with Force as Input'])  
plt.grid()

plt.subplot(313)
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), X[2])
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), r_st)
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), r_ac)
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), r_f)
plt.ylabel('r')
plt.xlabel('Time [s]')
plt.legend(['Data', 'Standard model', 'Model with Actuator Dynamics', 'Model with Force as Input'])
plt.grid()


######### error
plt.figure()
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), X[3] - eta_dot_nu_dot[3], linewidth = 0.8)
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), X[3]- du_ac_dyn_array, linewidth = 0.8)
plt.plot(np.arange(0, len(X_force[-1])*0.05, 0.05), X[3] - du_force_array, linewidth = 0.8)
plt.ylabel('Error in $\dot{u}$')
plt.legend(['Standard model', 'Model with Actuator Dynamics', 'Model with Force as Input'])
plt.grid()


plt.figure()
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), X[4]- eta_dot_nu_dot[4], linewidth = 0.8) 
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), X[4]- dv_ac_dyn_array, linewidth = 0.8)
plt.plot(np.arange(0, len(X_force[-1])*0.05, 0.05), X[4]- dv_force_array, linewidth = 0.8)
plt.ylabel('Error in $\dot{v}$')
plt.legend(['Standard model','Model with Actuator Dynamics', 'Model with Force as Input'])
plt.grid()

plt.figure()
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05),X[5]-  eta_dot_nu_dot[5], linewidth = 0.8) 
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), X[5]- dr_ac_dyn_array, linewidth = 0.8)
plt.plot(np.arange(0, len(X_force[-1])*0.05, 0.05),X[5]-  dr_force_array, linewidth = 0.8)
plt.ylabel('Error in $\dot{r}$')
plt.xlabel('Time [s]')
plt.legend(['Standard model', 'Model with Actuator Dynamics', 'Model with Force as Input'])
plt.grid()


plt.figure()
plt.plot(np.arange(0, len(X_force[-1])*0.05, 0.05), X[3] - du_force_array, linewidth = 0.5)
plt.ylabel('Error in $\dot{u}$')
plt.xlabel('Time [s]')
plt.grid()

plt.figure()
plt.plot(np.arange(0, len(X_force[-1])*0.05, 0.05), X[4]- dv_force_array, linewidth = 0.5)
plt.ylabel('Error in $\dot{v}$')
plt.xlabel('Time [s]')
plt.grid()

plt.figure()
plt.plot(np.arange(0, len(X_force[-1])*0.05, 0.05),X[5]-  dr_force_array, linewidth = 0.5)
plt.ylabel('Error in $\dot{r}$')
plt.xlabel('Time [s]')
plt.grid()






plt.show()