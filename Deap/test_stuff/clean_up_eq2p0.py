"""

equations with force as input

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
def plot_parts(func_list_rest, du = True, dv = False, dr = False):
	plt.figure()
	for i, sub_func in enumerate(func_list_rest):
		sub_function = str2func_force(sub_func)
		plt.subplot(len(func_list_rest)+1, 1, i+1)
		new_func = sub_function(X[0], X[1], X[2], X[6], X[7], X[8])
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




file = '/home/gislehalv/Master/Data/numpy_data_from_bag_force/all_bags_cut1.npy'
X = np.load(file)

#remove data with bucket < 95
index = []
for i in range(np.shape(X)[1]):
    if np.shape(X)[1] > i:
        if X[-2, i] < 95:
            index.append(i)
X = np.delete(X, index, 1)



du_str = '0.000264871203432643*fx -5.104448463432132e-07*fx*u**3 -2.773742056836302e-06*Abs(fz*v) + 2.649152675182239e-09*fx*Abs(fx) -0.00010529881090182071*r*v*Abs(fx) -0.037022654378220055*u**2 + 0.0020864289417205986*u**3 -1.722147024416803e-05*fx*r -9.437055733162733e-05*fx*u + 1.348464006791731e-05*fx*u**2 + 0.04778449754855396*r*u + 2.2407149473159294*r*v + 0.00947558535982701*u*v -0.0007198331382015782*u**2*v'
dv_str = '0.00017726146909316173*u**2 -2.9826827341053195e-07*fx*u -5.111840115947252e-05*fx*v + 4.192962483021315e-06*fz*u + 2.1305919250285774e-07*fz*v -0.20797031166711719*r*u + 0.27756501493808916*r*v -0.008344792948138165*u*v + 5.11254234557995e-06*u*Abs(fy) -1.0616675717356819e-05*v*Abs(fy)'
dr_str = '1.0842307778833146e-05*fx*v -5.1900734221065225e-09*fx*u**2 -5.619128822111664e-06*fy*v -2.1858712064244324e-08*fy*u**2 -4.099291394108764e-09*fx**2*r + 6.760385933163654e-12*fx**2*u -1.4636965662517697e-09*fx*fy*r + 3.040026213995778e-10*fx*fy*u + 1.9071315513956064e-06*fx*r*u -3.853593934102024e-06*fy*r*u + 2.5257972496607324e-05*fy'

du_str_split = ['0.000264*fx',' -5.1044e-07*fx*u**3 ','-2.7732e-06*Abs(fz*v) ','+ 2.649e-09*fx*Abs(fx) ','-0.000105*r*v*Abs(fx)',' -0.0370*u**2 ','+ 0.00208*u**3','-1.722e-05*fx*r ','-9.437e-05*fx*u',' + 1.348e-05*fx*u**2 ','+ 0.0477*r*u ','+ 2.2407*r*v ','+ 0.00947*u*v ','-0.000719*u**2*v']
dv_str_split = ['0.000177*u**2',' -2.9825e-07*fx*u ','-5.112e-05*fx*v ','+ 4.192e-06*fz*u ','+ 2.130e-07*fz*v ','-0.207*r*u ','+ 0.2776*r*v ','-0.00834*u*v ','+ 5.112e-06*u*Abs(fy) ','-1.061e-05*v*Abs(fy)']
dr_str_split = ['1.084e-05*fx*v ','-5.1905e-09*fx*u**2 ','-5.619e-06*fy*v',' -2.185e-08*fy*u**2 ','-4.099e-09*fx**2*r',' + 6.760e-12*fx**2*u',' -1.463e-09*fx*fy*r ','+ 3.040e-10*fx*fy*u ','+ 1.907e-06*fx*r*u',' -3.853e-06*fy*r*u',' + 2.525e-05*fy']



du_fun = str2func_force(du_str)
dv_fun = str2func_force(dv_str)
dr_fun = str2func_force(dr_str)


du_array = du_fun(X[0], X[1], X[2], X[6], X[7], X[8])
dv_array = dv_fun(X[0], X[1], X[2], X[6], X[7], X[8])
dr_array = dr_fun(X[0], X[1], X[2], X[6], X[7], X[8])


plot_parts(du_str_split, dr = False, du = True, dv = False)

plt.figure()
plt.plot(X[-1], X[4])
plt.plot(X[-1], dv_array)

plt.show()
