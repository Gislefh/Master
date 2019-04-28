from sympy import sympify, cos, sin, expand, collect, Lambda, lambdify, symbols
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/gislehalv/Master/scripts')
import my_lib
import math
import numpy as np

#du
eq_small = ' -5.015465314312387*r**2 + 0.010787502174784275*u**2 + 0.007307473954963339*sin(u)**2 + 3.135414581806233e-07*2*delta_t**2 -0.00021039428976226543*delta_t*v + 7.828050576367141e-05*delta_t*cos(u) + 0.884678750533654*r*v + 1.0541754076381125*r*cos(r) + 0.06584852953272069*r*cos(u) + 0.019297315056831632*u*v -0.26897900755277804*u*cos(r) + 0.01902230024683776*u*cos(u) + 0.2511746970208981*v*cos(r) -0.06029356401213615*v*sin(u) -0.37978015576637336*cos(r)*cos(u) + 0.31897338340622394*sin(u)*cos(r) + 0.0379281092537167*sin(u)*cos(u) + 4.966699849507045e-05*2*delta_t*cos(r) + 0.056841370146047066*2*r*u -0.13838713020550086*2*r*sin(u) -0.010981246609847131*2*u*sin(u) -0.0003283640961381819*3*delta_t*r -9.697605228418569e-06*3*delta_t*u -3.226831355353753e-05*3*delta_t*sin(u)'
eq_large = '-0.9696637145243585*r**2 + 0.179249347989753*cos(r)**2 + 5.787880112739785e-06*2*delta_t**2 + 1.533076945990353e-10*2*delta_t**3 + 0.004429827697141425*3*u**2 + 7.35986652515841e-05*delta_n*delta_t -0.023620994053089817*delta_n*r + 0.001419221037489038*delta_n*u + 0.0007326784442209799*delta_n*v**2 -0.007136236361716897*delta_n*cos(r) + 0.0015101271309863051*delta_n*sin(u) -0.002177044053496502*delta_t*v -1.4994603690887372e-07*delta_n**2*delta_t -0.00841111253521376*delta_t*r**2 + 2.631125512575494e-06*delta_t*v**2 + 1.2932939152804357e-06*delta_t*cos(u) -1.7103870304708835*r*v + 1.9457613913811185e-10*delta_t**3*r + 0.029841400483348934*r*cos(u) -0.3949309287918368*r*sin(u) -0.02362472325643239*u*v + 0.02311639993614989*u*cos(u) -2.553735653360034e-05*delta_n**2*v + 0.36300781961324446*v*cos(r) -0.06268221180379241*v*sin(u) -1.1599446437804417e-05*delta_t**2*cos(r) + 1.2858457062806525e-08*delta_t**2*cos(u) -0.3082408991829766*cos(r)*cos(u) + 0.1892810788678787*sin(u)*cos(r) + 0.032667614181393745*sin(u)*cos(u) -1.5075098453887215e-05*2*delta_t*sin(u) + 5.09837795939494e-09*2*delta_t**2*r + 1.1102173809777014*2*r*cos(r) + 9.932171084656732e-10*3*delta_n*delta_t**2 -0.00510914349457825*3*delta_t*r + 0.00012376751312515033*3*delta_t*cos(r) -1.5673898979292575e-08*3*delta_t**2*u -0.0063906987412881144*3*u*sin(u) + 2.404593331715261e-09*3*delta_t**2*v -0.0065245221924215*4*r*u -0.08733620661522767*4*u*cos(r) + 1.3325460756930668e-05*5*delta_t*u + 0.00013016051861569622*delta_n*delta_t*r -7.454785443714029e-05*delta_n*delta_t*cos(r) -1.2191768236505598e-07*delta_n*delta_t*cos(u) -0.044303868261309276*delta_n*r*v -8.05191869707178e-08*delta_n*delta_t**2*r + 0.012345894047944839*delta_n*v*cos(r) + 0.00027109332765196115*delta_n*v*cos(u) + 5.679853469048313e-05*delta_t*r*u + 0.004779752625154288*delta_t*r*v + 0.01350911186180781*delta_t*r*cos(r) -7.137030082615681e-05*delta_t*r*sin(u) + 0.0019763690279752666*delta_t*v*cos(r) -6.805563755093308e-07*delta_t*v*cos(u) -1.4623922773848719e-06*delta_t**2*r*v -3.3325811011394174e-07*3*delta_n*delta_t*u + 7.49072688681629e-05*3*delta_n*u*v + 5.106784602172676e-06*3*delta_t*u*v -3.1468583414151832e-06*4*delta_n*delta_t*v + 3.335693556949515e-05*delta_n*delta_t*r*v'

def str2func(exp_str):
	u, v, r, delta_t, delta_n = symbols('u v r delta_t delta_n')
	f = lambdify((u,v,r, delta_t, delta_n), exp_str, 'numpy')
	return f

func_small = str2func(eq_small)
func_large = str2func(eq_large)


###-- get data
X1 = np.load('/home/gislehalv/Master/Data/numpy_data_from_bag/' + 'bag1_025'+'.npy')
X2 = np.load('/home/gislehalv/Master/Data/numpy_data_from_bag/' + 'bag2_025'+'.npy')
X3 = np.load('/home/gislehalv/Master/Data/numpy_data_from_bag/' + 'bag3_025'+'.npy')
X4 = np.load('/home/gislehalv/Master/Data/numpy_data_from_bag/' + 'bag4_025'+'.npy')


#fix time 
X2[-1] = X2[-1] + X1[-1, -1]
X3[-1] = X3[-1] + X2[-1, -1]
X4[-1] = X4[-1] + X3[-1, -1]

#melt all the data together
X = np.concatenate((X1,X2,X3,X4),axis = 1)

index = []
for i in range(np.shape(X)[1]):
	if np.shape(X)[1] > i:
		if X[-2, i] < 95:
			index.append(i)

X = np.delete(X, index, 1)

u = X[0]
v = X[1]
r = X[2]
du = X[3]
delta_t = X[-4]
delta_n = X[-3]
y = X[5]        #  <----dr btw
time = X[-1]
delta_n[delta_n > 27] = 27 #remove error in the data
delta_n[delta_n < -27] = -27

plt.figure()
plt.plot(time, func_small(u, v, r, delta_t, delta_n))
plt.plot(time, func_large(u, v, r, delta_t, delta_n))
plt.plot(time, du)
plt.legend(['pred small', 'pred large', 'du'])
plt.show()
