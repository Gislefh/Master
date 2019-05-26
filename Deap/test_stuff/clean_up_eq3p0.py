"""

equations with contorl inputs, WITH  delta_n in RADS!!!!

"""
from sympy import sympify, cos, sin, expand, collect, Lambda, lambdify, symbols
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/gislehalv/Master/scripts')
import my_lib
import math
import numpy as np


##takes in string -> gives out the function for the string
def str2func(exp_str):
	u, v, r, delta_t, delta_n = symbols('u v r delta_t delta_n')
	f = lambdify((u,v,r, delta_t, delta_n), exp_str, 'numpy')
	return f

def least_sq(func_list):
	F_list  = []
	for func in func_list:
		F_list.append(str2func(func))
	F = np.zeros((len(y), len(F_list)))

	for i, function in enumerate(F_list):
		F[:,i] = np.squeeze(function(u, v, r, delta_t, delta_n))

	F_trans = np.transpose(F)
	try:
		p = np.dot(np.linalg.inv(np.dot(F_trans,F)),np.dot(F_trans,y))  
	except:
		print('Singular Matrix for: ', individual)
		exit()
		mse = 1000 # large number
		return(mse,)

	tot_func = np.zeros((len(y)))

	for i, func in enumerate(F_list):
		tot_func = np.add(tot_func, p[i]*func(u, v, r, delta_t, delta_n))

	return tot_func

#removes parts that are smaller than the threshold
def remove_insig_parts(eq_list):
	func_list_rest  = []
	for sub_f in sub_func_list_small:
		sub_function = str2func(sub_f)
		if np.amax(np.abs(sub_function(u,v,r,delta_t,delta_n))) < 1e-5: 
			print('removed:', sub_f,'  the largest absolute value were:', np.amax(np.abs(sub_function(u,v,r,delta_t,delta_n))))
		else:
			func_list_rest.append(sub_f)
	return func_list_rest

# plots all the parts seperatly
def plot_parts(func_list_rest, du = False, dv = False, dr = False):
	plt.figure()
	for i, sub_func in enumerate(func_list_rest):
		sub_function = str2func(sub_func)
		plt.subplot(len(func_list_rest)+1, 1, i+1)
		new_func = sub_function(X[0], X[1], X[2], X[7], X[8])
		plt.plot(X[-1], new_func, linewidth = 0.8)
		plt.legend([sub_func])
		
		plt.grid()
	plt.subplot(len(func_list_rest)+1, 1, i+2)
	
	if du:
		plt.plot(X[-1], X[3], 'r', linewidth = 0.8)
	if dv:
		plt.plot(X[-1], X[4], 'r', linewidth = 0.8)
	if dr:
		plt.plot(X[-1], X[5], 'r', linewidth = 0.8)
	
	plt.grid()
	plt.xlabel('Time [s]')
	plt.ylabel('Ground Truth')




def plot_parts_w_sol(func_list_rest, gt, xlim = [], ylim = []):
	plt.figure()
	for i, sub_func in enumerate(func_list_rest):
		sub_function = str2func(sub_func)
		plt.subplot(len(func_list_rest), 1, i+1)
		new_func = sub_function(u,v,r,delta_t,delta_n)
		plt.plot(time, new_func)
		plt.plot(time, y)
		plt.legend([sub_func])
		plt.grid()
		if xlim:
			plt.xlim(xlim[0], xlim[1])
		if ylim:
			plt.ylim([ylim[0], ylim[1]])


	plt.xlabel('Time [s]')



def str2func_force(exp_str):
    u, v, r, fx, fy, fz = symbols('u v r fx, fy, fz')
    f = lambdify((u,v,r, fx, fy, fz), exp_str, 'numpy')
    return f




### ---load data---
path = '/home/gislehalv/Master/scripts/standard_model/'


X = np.load(path +'Data_cut01.npy')

#remove data with bucket < 95
index = []
for i in range(np.shape(X)[1]):
    if np.shape(X)[1] > i:
        if X[-2, i] < 95:
            index.append(i)
X = np.delete(X, index, 1)



X[8][X[8] > 27 * np.pi / 180] =27 * np.pi / 180
X[8][X[8] < -27 * np.pi / 180] = -27 * np.pi / 180


du_str = '-6.402e-07*delta_t**2*Abs(r) -1.751e-07*delta_t**2*Abs(v) + 9.536e-14*delta_t**3*Abs(delta_t) -0.000289*delta_t*u*Abs(r) + 5.812e-05*delta_t*u*Abs(v) + 1.622e-08*delta_t*u**2*Abs(delta_t) + 1.675e-07*delta_t**2*u*Abs(r) -1.772e-08*delta_t**2*u*Abs(v) + 1.898e-14*delta_t**3*u*Abs(delta_t) -7.242e-12*delta_t**2*u**2*Abs(delta_t) -1.192e-11*2*delta_t**2*u*Abs(delta_t) -0.155*u -0.000137*delta_t*u + 0.000855*delta_t'
du_str_split =  ['-6.402e-07*delta_t**2*Abs(r) ','-1.751e-07*delta_t**2*Abs(v) ','+ 9.536e-14*delta_t**3*Abs(delta_t) ','-0.000289*delta_t*u*Abs(r) ','+ 5.812e-05*delta_t*u*Abs(v) ','+ 1.622e-08*delta_t*u**2*Abs(delta_t)',' + 1.675e-07*delta_t**2*u*Abs(r) ','-1.772e-08*delta_t**2*u*Abs(v)',' + 1.898e-14*delta_t**3*u*Abs(delta_t) ','-7.242e-12*delta_t**2*u**2*Abs(delta_t) ','-1.192e-11*2*delta_t**2*u*Abs(delta_t) ','-0.155*u ','-0.000137*delta_t*u ','+ 0.000855*delta_t']

dv_str = ' -0.120*delta_n*u -0.0424*delta_n*v + 2.695e-06*delta_t*u -0.000294*delta_t*v -0.208*r*u -0.00694*r*v -0.00294*u*Abs(v) + 0.0188*v*Abs(v)'
dv_str_split = [' -0.120*delta_n*u ','-0.0424*delta_n*v ','+ 2.695e-06*delta_t*u ','-0.000294*delta_t*v ','-0.208*r*u ','-0.00694*r*v ','-0.00294*u*Abs(v)',' + 0.0188*v*Abs(v)']

dr_str = '0.00512*delta_n**3 + 0.00560*v**3 + 1.005e-07*delta_n*delta_t**2 + 0.00160*delta_n**2*u -0.00150*u*v**2 + 7.503e-08*delta_t**2*v -3.438e-05*2*delta_n**2*delta_t + 4.645e-06*2*delta_t*v**2 + 0.0372*3*delta_n*v**2 + 0.0463*3*delta_n**2*v + 3.122e-06*delta_n*delta_t*u -6.894e-06*delta_t*u*v + 0.00348*2*delta_n*u*v -9.361e-06*4*delta_n*delta_t*v'
dr_str_split = ['0.00512*delta_n**3 ','+ 0.00560*v**3 ','+ 1.005e-07*delta_n*delta_t**2 ','+ 0.00160*delta_n**2*u ','-0.00150*u*v**2 ','+ 7.503e-08*delta_t**2*v ','-3.438e-05*2*delta_n**2*delta_t ','+ 4.645e-06*2*delta_t*v**2',' + 0.0372*3*delta_n*v**2',' + 0.0463*3*delta_n**2*v',' + 3.122e-06*delta_n*delta_t*u ','-6.894e-06*delta_t*u*v',' + 0.00348*2*delta_n*u*v',' -9.361e-06*4*delta_n*delta_t*v']

du_fun = str2func(du_str)
dv_fun = str2func(dv_str)
dr_fun = str2func(dr_str)


du_array = du_fun(X[0], X[1], X[2], X[7], X[8])
dv_array = dv_fun(X[0], X[1], X[2], X[7], X[8])
dr_array = dr_fun(X[0], X[1], X[2], X[7], X[8])


plot_parts(du_str_split, du = True)
plot_parts(dv_str_split, dv = True)
plot_parts(dr_str_split, dr = True)

plt.figure()
plt.plot(X[-1], X[4])
plt.plot(X[-1], dv_array)

plt.show()
