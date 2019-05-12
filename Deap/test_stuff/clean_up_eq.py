"""
- clean up eq

- find the impact of the parts

- find new LS pararmerters ?
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
def plot_parts(func_list_rest):
	plt.figure()
	for i, sub_func in enumerate(func_list_rest):
		sub_function = str2func(sub_func)
		plt.subplot(len(func_list_rest), 1, i+1)
		new_func = sub_function(u,v,r,delta_t,delta_n)
		plt.plot(time, new_func)
		plt.legend([sub_func])
		plt.grid()

		if  i == 0:
			tot_func = new_func
		else:
			tot_func = np.add(tot_func, new_func)

	plt.xlabel('Time [s]')
	plt.figure()
	plt.plot(time, tot_func)
	plt.plot(time, sol)
	plt.plot(time, y)
	plt.grid()
	plt.legend(['tot_func', 'orig' ,'GT'])
	


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

		# if  i == 0:
		# 	tot_func = new_func
		# else:
		# 	tot_func = np.add(tot_func, new_func)

	plt.xlabel('Time [s]')
	# plt.figure()
	# plt.plot(time, tot_func)
	# plt.plot(time, sol)
	# plt.plot(time, y)
	#plt.grid()
	#plt.legend(['tot_func', 'orig' ,'GT'])
	

###-- get data
if 1:
	X1 = np.load('/home/gislehalv/Master/Data/numpy_data_from_bag/' + 'bag1_025'+'.npy')
	X2 = np.load('/home/gislehalv/Master/Data/numpy_data_from_bag/' + 'bag2_025'+'.npy')
	X3 = np.load('/home/gislehalv/Master/Data/numpy_data_from_bag/' + 'bag3_025'+'.npy')
	X4 = np.load('/home/gislehalv/Master/Data/numpy_data_from_bag/' + 'bag4_025'+'.npy')


	#fix time 
	X2[-1] = X2[-1] + X1[-1, -1]
	X3[-1] = X3[-1] + X2[-1, -1]
	#X4[-1] = X4[-1] + X3[-1, -1]

	#melt all the data together
	X = np.concatenate((X1,X2,X3),axis = 1)

	#remove data with bucket < 95
	index = []
	for i in range(np.shape(X)[1]):
		if np.shape(X)[1] > i:
			if X[-2, i] < 95:
				index.append(i)
	X = np.delete(X, index, 1)
	
	# X_val = np.concatenate((X[:,11511:18500], X[:,27912:32693], X[:,56531:60714], X[:,70475:74193]),axis = 1)
	# index = list(range(11511,18500)) + list(range(27912,32693)) +list(range(56531,60714)) + list(range(70475,74193))
	# X = np.delete(X, index, 1)

	u = X[0]
	v = X[1]
	r = X[2]
	delta_t = X[-4]
	delta_n = X[-3]
	y = X[3]        #  <----dv btw
	time = X[-1]
	delta_n[delta_n > 27] = 27 #remove error in the data
	delta_n[delta_n < -27] = -27


du_ = True
dv_ = False
dr_ = False
###  ---------  du -----------
if du_:
	orig_str = '-5.015465314312387*r**2 + 0.010787502174784275*u**2 + 0.007307473954963339*sin(u)**2 + 3.135414581806233e-07*2*delta_t**2 -0.00021039428976226543*delta_t*v + 7.828050576367141e-05*delta_t*cos(u) + 0.884678750533654*r*v + 1.0541754076381125*r*cos(r) + 0.06584852953272069*r*cos(u) + 0.019297315056831632*u*v -0.26897900755277804*u*cos(r) + 0.01902230024683776*u*cos(u) + 0.2511746970208981*v*cos(r) -0.06029356401213615*v*sin(u) -0.37978015576637336*cos(r)*cos(u) + 0.31897338340622394*sin(u)*cos(r) + 0.0379281092537167*sin(u)*cos(u) + 4.966699849507045e-05*2*delta_t*cos(r) + 0.056841370146047066*2*r*u -0.13838713020550086*2*r*sin(u) -0.010981246609847131*2*u*sin(u) -0.0003283640961381819*3*delta_t*r -9.697605228418569e-06*3*delta_t*u -3.226831355353753e-05*3*delta_t*sin(u)'
	orig_str_split = ['-5.015465314312387*r**2 ','+ 0.010787502174784275*u**2 ','+ 0.007307473954963339*sin(u)**2 ','+ 3.135414581806233e-07*2*delta_t**2 ','-0.00021039428976226543*delta_t*v ','+ 7.828050576367141e-05*delta_t*cos(u) ','+ 0.884678750533654*r*v ','+ 1.0541754076381125*r*cos(r)',' + 0.06584852953272069*r*cos(u) ','+ 0.019297315056831632*u*v ','-0.26897900755277804*u*cos(r) ','+ 0.01902230024683776*u*cos(u)',' + 0.2511746970208981*v*cos(r)',' -0.06029356401213615*v*sin(u) ','-0.37978015576637336*cos(r)*cos(u) ','+ 0.31897338340622394*sin(u)*cos(r) ','+ 0.0379281092537167*sin(u)*cos(u)',' + 4.966699849507045e-05*2*delta_t*cos(r) ','+ 0.056841370146047066*2*r*u ','-0.13838713020550086*2*r*sin(u)',' -0.010981246609847131*2*u*sin(u) ','-0.0003283640961381819*3*delta_t*r ','-9.697605228418569e-06*3*delta_t*u ',' -3.226831355353753e-05*3*delta_t*sin(u)']

	orig_str_reduced = '-5.015465314312387*r**2 + 0.010787502174784275*u**2 + 3.135414581806233e-07*2*delta_t**2 -0.00021039428976226543*delta_t*v + 7.828050576367141e-05*delta_t*cos(u) + 0.884678750533654*r*v + 1.0541754076381125*r*cos(r) + 0.019297315056831632*u*v -0.26897900755277804*u*cos(r) + 0.01902230024683776*u*cos(u) + 0.2511746970208981*v*cos(r) -0.06029356401213615*v*sin(u) -0.37978015576637336*cos(r)*cos(u) + 0.31897338340622394*sin(u)*cos(r) + 4.966699849507045e-05*2*delta_t*cos(r) + 0.056841370146047066*2*r*u -0.13838713020550086*2*r*sin(u) -0.010981246609847131*2*u*sin(u) -0.0003283640961381819*3*delta_t*r -9.697605228418569e-06*3*delta_t*u -3.226831355353753e-05*3*delta_t*sin(u)'

	orig_str_no_rsq = ' 0.010787502174784275*u**2 + 3.135414581806233e-07*2*delta_t**2 -0.00021039428976226543*delta_t*v + 7.828050576367141e-05*delta_t*cos(u) + 0.884678750533654*r*v + 1.0541754076381125*r*cos(r) + 0.019297315056831632*u*v -0.26897900755277804*u*cos(r) + 0.01902230024683776*u*cos(u) + 0.2511746970208981*v*cos(r) -0.06029356401213615*v*sin(u) -0.37978015576637336*cos(r)*cos(u) + 0.31897338340622394*sin(u)*cos(r) + 4.966699849507045e-05*2*delta_t*cos(r) + 0.056841370146047066*2*r*u -0.13838713020550086*2*r*sin(u) -0.010981246609847131*2*u*sin(u) -0.0003283640961381819*3*delta_t*r -9.697605228418569e-06*3*delta_t*u -3.226831355353753e-05*3*delta_t*sin(u)'
	orig_str_no_cos_sin = '-5.015465314312387*r**2 + 0.010787502174784275*u**2 + 3.135414581806233e-07*2*delta_t**2 -0.00021039428976226543*delta_t*v + 7.828050576367141e-05*delta_t*cos(u) + 0.884678750533654*r*v + 1.0541754076381125*r*cos(r) + 0.019297315056831632*u*v -0.26897900755277804*u*cos(r) + 0.01902230024683776*u*cos(u) + 0.2511746970208981*v*cos(r) -0.06029356401213615*v*sin(u) + 4.966699849507045e-05*2*delta_t*cos(r) + 0.056841370146047066*2*r*u -0.13838713020550086*2*r*sin(u) -0.010981246609847131*2*u*sin(u) -0.0003283640961381819*3*delta_t*r -9.697605228418569e-06*3*delta_t*u -3.226831355353753e-05*3*delta_t*sin(u)'
	orig_str_no_cosr = '-5.015465314312387*r**2 + 0.010787502174784275*u**2 + 3.135414581806233e-07*2*delta_t**2 -0.00021039428976226543*delta_t*v + 7.828050576367141e-05*delta_t*cos(u) + 0.884678750533654*r*v + 1.0541754076381125*r + 0.019297315056831632*u*v -0.26897900755277804*u + 0.01902230024683776*u*cos(u) + 0.2511746970208981*v -0.06029356401213615*v*sin(u) -0.37978015576637336*cos(u) + 0.31897338340622394*sin(u) + 4.966699849507045e-05*2*delta_t + 0.056841370146047066*2*r*u -0.13838713020550086*2*r*sin(u) -0.010981246609847131*2*u*sin(u) -0.0003283640961381819*3*delta_t*r -9.697605228418569e-06*3*delta_t*u -3.226831355353753e-05*3*delta_t*sin(u)'
	orig_str_no_cosr_split = ['-5.015465314312387*r**2',' + 0.010787502174784275*u**2',' + 3.135414581806233e-07*2*delta_t**2 ','-0.00021039428976226543*delta_t*v',' + 7.828050576367141e-05*delta_t*cos(u) ','+ 0.884678750533654*r*v',' + 1.0541754076381125*r ','+ 0.019297315056831632*u*v ','-0.26897900755277804*u ','+ 0.01902230024683776*u*cos(u)',' + 0.2511746970208981*v ','-0.06029356401213615*v*sin(u) ','-0.37978015576637336*cos(u) ','+ 0.31897338340622394*sin(u)',' + 4.966699849507045e-05*2*delta_t ','+ 0.056841370146047066*2*r*u',' -0.13838713020550086*2*r*sin(u)',' -0.010981246609847131*2*u*sin(u)',' -0.0003283640961381819*3*delta_t*r',' -9.697605228418569e-06*3*delta_t*u ','-3.226831355353753e-05*3*delta_t*sin(u)']

	### functions
	func = str2func(orig_str)
	func_red = str2func(orig_str_reduced)
	func_nor = str2func(orig_str_no_rsq)
	func_nocossin = str2func(orig_str_no_cos_sin)
	func_no_cosr = str2func(orig_str_no_cosr)

	### arrays from funcs
	sol = func(u,v,r,delta_t,delta_n)
	sol_red = func_red(u,v,r,delta_t,delta_n)
	sol_nor = func_nor(u,v,r,delta_t,delta_n)
	sol_nocossin = func_nocossin(u,v,r,delta_t,delta_n)
	sol_no_cosr = func_no_cosr(u,v,r,delta_t,delta_n)

	# plot str parts
	#plot_parts(orig_str_no_cosr_split)
	




	# plt.figure()
	# #plt.subplot(211)
	# plt.plot(time, sol)
	# plt.plot(time, sol_no_cosr)
	# plt.plot(time, y)
	# plt.grid()
	# plt.ylabel('du [m/s^2]')
	# plt.xlabel('Time [s]')
	# plt.legend(['original','no cos(r)' ,'ground truth'])
	#plt.subplot(212)
	#plt.plot(time, sol - y)
	#plt.plot(time, sol_nocossin - y)
	#plt.legend(['orig error', 'error no cos sin'])
	#plt.grid()

	# plt.figure()
	# plt.plot(time, r)
	# plt.grid()
	# plt.ylabel('r')
	# plt.xlabel('Time [s]')


	plt.figure()
	plt.plot(time, sol_no_cosr)
	plt.plot(time, y)
	plt.legend(['Model', 'Data'])
	plt.xlabel('Time [s]')
	plt.ylabel('du [m/s^2]')
	plt.grid()
	plt.show()

#### --------- dv ------------ 
if dv_:
	### str
	orig_str = '-3.952212504979791e-06*delta_n**3*u -4.323806251907643e-06*delta_n**3*v + 3.633763321792768e-06*delta_n**2*u**2 -1.831870024171225e-07*delta_n*delta_t*u**2 + 0.0003823435101753467*delta_n*u**2*sin(cos(r)) -2.1912674505216164e-09*delta_n**2*delta_t*u -8.612870897527819e-07*delta_n**2*delta_t*v -3.3142056104988395e-09*delta_t*u**2*sin(cos(r)) + 7.719118258445558e-05*delta_n**2*u*v + 5.402037395386755e-06*delta_n**2*u*sin(cos(r)) + 0.0005809985939579979*delta_n**2*v*sin(cos(r)) + 6.585244046079131e-07*delta_n*delta_t*u*v -7.821381053574049e-07*delta_n*delta_t*u*sin(cos(r)) + 6.313586874164905e-06*delta_n*delta_t*v*sin(cos(r)) -0.0023077638335907077*delta_n*u*v*sin(cos(r)) -1.4485249323632313e-05*delta_t*u*v*sin(cos(r))'
	orig_str_split = ['-3.952212504979791e-06*delta_n**3*u ','-4.323806251907643e-06*delta_n**3*v ','+ 3.633763321792768e-06*delta_n**2*u**2 ','-1.831870024171225e-07*delta_n*delta_t*u**2 ','+ 0.0003823435101753467*delta_n*u**2*sin(cos(r))',' -2.1912674505216164e-09*delta_n**2*delta_t*u',' -8.612870897527819e-07*delta_n**2*delta_t*v',' -3.3142056104988395e-09*delta_t*u**2*sin(cos(r))',' + 7.719118258445558e-05*delta_n**2*u*v',' + 5.402037395386755e-06*delta_n**2*u*sin(cos(r))',' + 0.0005809985939579979*delta_n**2*v*sin(cos(r)) ','+ 6.585244046079131e-07*delta_n*delta_t*u*v ','-7.821381053574049e-07*delta_n*delta_t*u*sin(cos(r)) ','+ 6.313586874164905e-06*delta_n*delta_t*v*sin(cos(r)) ','-0.0023077638335907077*delta_n*u*v*sin(cos(r)) ',' -1.4485249323632313e-05*delta_t*u*v*sin(cos(r))']
	new_str = '-1.7834290681198802e-07*delta_t**2*v + 6.706306888210264e-10*delta_t**2*Abs(u) + 1.076964429624133e-06*delta_n*delta_t*v -1.5462579978451662e-06*delta_n*delta_t*Abs(u) + 0.008807113767162776*delta_n*r*v + 0.0061839222365966135*delta_n*r*Abs(u) + 0.0002921184004195739*delta_t*r*v -9.297713363841411e-05*delta_t*r*Abs(u)'


	str_no_low_vals = '-3.952212504979791e-06*delta_n**3*u -4.323806251907643e-06*delta_n**3*v + 3.633763321792768e-06*delta_n**2*u**2 -1.831870024171225e-07*delta_n*delta_t*u**2 + 0.0003823435101753467*delta_n*u**2*sin(cos(r)) -8.612870897527819e-07*delta_n**2*delta_t*v + 7.719118258445558e-05*delta_n**2*u*v  + 0.0005809985939579979*delta_n**2*v*sin(cos(r)) + 6.585244046079131e-07*delta_n*delta_t*u*v -7.821381053574049e-07*delta_n*delta_t*u*sin(cos(r)) + 6.313586874164905e-06*delta_n*delta_t*v*sin(cos(r)) -0.0023077638335907077*delta_n*u*v*sin(cos(r)) -1.4485249323632313e-05*delta_t*u*v*sin(cos(r))'
	str_no_sincos = '-3.952212504979791e-06*delta_n**3*u -4.323806251907643e-06*delta_n**3*v + 3.633763321792768e-06*delta_n**2*u**2 -1.831870024171225e-07*delta_n*delta_t*u**2 + 0.0003823435101753467*delta_n*u**2*0.82 -8.612870897527819e-07*delta_n**2*delta_t*v + 7.719118258445558e-05*delta_n**2*u*v  + 0.0005809985939579979*delta_n**2*v*0.82 + 6.585244046079131e-07*delta_n*delta_t*u*v -7.821381053574049e-07*delta_n*delta_t*u*0.82 + 6.313586874164905e-06*delta_n*delta_t*v*0.82 -0.0023077638335907077*delta_n*u*v*0.82 -1.4485249323632313e-05*delta_t*u*v*0.82'
	str_no_sincos_split = ['-3.952212504979791e-06*delta_n**3*u',' -4.323806251907643e-06*delta_n**3*v',' + 3.633763321792768e-06*delta_n**2*u**2 ','-1.831870024171225e-07*delta_n*delta_t*u**2 ','+ 0.0003823435101753467*delta_n*u**2*0.82 ','-8.612870897527819e-07*delta_n**2*delta_t*v ','+ 7.719118258445558e-05*delta_n**2*u*v  ','+ 0.0005809985939579979*delta_n**2*v*0.82 ','+ 6.585244046079131e-07*delta_n*delta_t*u*v ','-7.821381053574049e-07*delta_n*delta_t*u*0.82 ','+ 6.313586874164905e-06*delta_n*delta_t*v*0.82',' -0.0023077638335907077*delta_n*u*v*0.82 ','-1.4485249323632313e-05*delta_t*u*v*0.82']
	
	str_remove_stuff = '-3.952212504979791e-06*delta_n**3*u -4.323806251907643e-06*delta_n**3*v  + 0.0003823435101753467*delta_n*u**2*sin(cos(r)) -8.612870897527819e-07*delta_n**2*delta_t*v + 7.719118258445558e-05*delta_n**2*u*v  + 0.0005809985939579979*delta_n**2*v*sin(cos(r)) + 6.585244046079131e-07*delta_n*delta_t*u*v -7.821381053574049e-07*delta_n*delta_t*u*sin(cos(r)) + 6.313586874164905e-06*delta_n*delta_t*v*sin(cos(r)) -0.0023077638335907077*delta_n*u*v*sin(cos(r)) -1.4485249323632313e-05*delta_t*u*v*sin(cos(r))'

	##funcs
	new_func = str2func(new_str)
	orig_func = str2func(orig_str)
	func_no_low = str2func(str_no_low_vals)
	func_no_sincos = str2func(str_no_sincos)
	func_remove_stuff = str2func(str_remove_stuff)
	


	#arrays 
	sol = orig_func(u,v,r,delta_t,delta_n)
	sol_no_low = func_no_low(u,v,r,delta_t,delta_n)
	sol_no_sincos = func_no_sincos(u,v,r,delta_t,delta_n)
	sol_remove_suff = func_remove_stuff(u,v,r,delta_t,delta_n)
	new_sol = new_func(u,v,r,delta_t,delta_n)


	#error
	# plt.figure()
	# plt.plot(time, y - sol)
	# plt.grid()
	# plt.xlabel('Time [s]')
	# plt.ylabel('error')
 
	#plot no low vals - EQUAL
	# plt.figure()
	# plt.plot(time, sol)
	# plt.plot(time, sol_no_low)
	# plt.plot(time, y)
	# plt.legend(['orig', 'no low', 'GT'])
	# plt.show()

	# plot no sin(cos(r))
	# plt.figure()
	# plt.plot(time, sol)
	# plt.plot(time, sol_no_sincos)
	# plt.plot(time, y)
	# plt.grid()
	# plt.legend(['orig', 'no sin(cos(r))', 'GT'])
	# plt.show()

	#plot - removed delta_n^2*u^2 and delta_n*delta_t*u^2 - looks shit
	# plt.figure()
	# plt.plot(time, sol)
	# plt.plot(time, sol_remove_suff)
	# plt.plot(time, y)
	# plt.grid()
	# plt.legend(['orig', 'removed stuff', 'GT'])
	# plt.show()


	#plot solution
	plt.figure()
	plt.plot(time, sol)
	plt.plot(time, y)
	plt.plot(time, new_sol)
	plt.grid()
	plt.legend(['old sol', 'gt', ' new sol'])
	#plt.show()

	# #plot_parts(str_no_sincos_split)
	# new_param = least_sq(str_no_sincos_split)


	# plt.plot(time, new_param)

	# plt.legend(['Model', 'Ground Truth', 'New params from LS'])
	# plt.xlabel('Time [s]')
	# plt.ylabel('dr [m/s^2]')
	# plt.show()

#### --------- dr ------------ 
if dr_:
	##  string
	str_simple = '2.0876035809881317e-06*delta_n*delta_t + 6.309528565753061e-05*delta_n*u + 7.80439835159352e-05*delta_t*v + 0.0925254710170566*r*v**2 -0.006498071493868284*u*v + 0.0046659143440125235*delta_n*r*v'
	str_complex = '8.676930433282843e-10*2*delta_t**2 + 0.0012570249155565497*delta_n*v**2 -3.5791825989844134e-07*delta_t*u -1.8080475412948896e-05*delta_t*v + 4.90666449017099e-06*delta_t*v**2 + 4.0035352353309594e-07*delta_n**2*u + 4.968472752402066e-05*delta_n**2*v + 7.414440664621671e-10*2*delta_n*delta_t**2 -7.66056697764632e-09*2*delta_n**2*delta_t + 3.512565795405815e-08*2*delta_t**2*v -6.772198557462273e-09*delta_n*delta_t*u + 0.00020528245018887196*delta_n*u*v -6.381357167124244e-06*delta_t*u*v -2.729608978246139e-07*3*delta_n*delta_t*v'
	new_str = '  + 1.8100389677413597e-12*delta_t**3 + 1.3724567990403322e-09*delta_n*delta_t**2 -1.0985347071361126e-07*delta_t**2*r -6.446791375947852e-10*delta_t**2*Abs(delta_n) -1.7538736385951493e-08*2*delta_t**2*v + 3.527724015841834e-11*2*delta_t**3*v -1.2989591563981094e-13*2*delta_n**2*delta_t**3 + 3.0946320086240535e-12*2*delta_n**3*delta_t**2 + 7.280580161139506e-13*3*delta_n*delta_t**3 + 1.8106630192792778e-10*3*delta_n**2*delta_t**2 + 7.4359945722531225e-09*4*delta_t**2*v**2 -2.3113108405113357e-08*delta_n*delta_t*Abs(delta_n) -4.159867174462178e-10*delta_n*delta_t**2*u + 5.6595357816064534e-06*delta_t*r*Abs(delta_n) -1.6886707882771762e-07*delta_n**2*delta_t*u -2.5434827509804167e-09*delta_n**3*delta_t*u -7.018686972119614e-09*delta_t**2*u*v + 7.651807471533269e-11*delta_n**2*delta_t**2*u + 2.5949116461733153e-06*delta_n**2*u*Abs(delta_n) -8.300368028946997e-13*2*delta_n*delta_t**3*v -9.98397819191713e-11*2*delta_n*delta_t**2*Abs(delta_n) -2.526494753077736e-06*2*delta_t*u*v**2 + 1.1081632982249079e-06*2*delta_t*v*Abs(delta_n) -3.200754444927704e-09*2*delta_n**2*delta_t*Abs(delta_n) -6.50017268972543e-08*2*delta_t**2*r*v -3.2396901556092847e-10*2*delta_n**2*delta_t**2*r + 0.000252334378744945*2*u*v**2*Abs(delta_n) + 3.182019457982639e-10*2*delta_t**2*v*Abs(delta_n) + 2.297145185902325e-09*3*delta_n*delta_t**2*r + 1.5904285959210878e-10*4*delta_n*delta_t**2*v**2 -7.604926706601475e-07*4*delta_t*v**2*Abs(delta_n) -2.969725374716412e-11*6*delta_n**2*delta_t**2*v + 6.06926543052777e-10*8*delta_n*delta_t**2*v -1.051890095633254e-06*delta_n*delta_t*r*u + 8.714687163112072e-08*delta_n*delta_t*u*Abs(delta_n) + 8.027410464577335e-06*delta_n*r*u*Abs(delta_n) + 4.6541189069514634e-10*delta_n*delta_t**2*u*v + 2.2541360046706262e-05*delta_t*r*u*v + 1.1528058408885149e-07*delta_n**2*delta_t*r*u + 4.7337525317626365e-08*delta_t*u*v*Abs(delta_n) -0.000598723031867282*r*u*v*Abs(delta_n) -1.5563455348796903e-07*2*delta_n*delta_t*r*Abs(delta_n) -2.427463672171879e-08*2*delta_n*delta_t*u*v**2 + 9.58327909283671e-10*2*delta_n*delta_t**2*r*v + 2.3692812317169004e-06*2*delta_t*r*v*Abs(delta_n) -4.564591851159667e-07*3*delta_n*delta_t*u*v + 1.2149958104324104e-05*3*delta_n*u*v*Abs(delta_n) + 1.0016982615552704e-08*3*delta_n**2*delta_t*u*v -2.6977864531837245e-08*6*delta_n*delta_t*v*Abs(delta_n) -8.664364428341315e-07*delta_n*delta_t*r*u*v'
	new_str_simple = '-1.8790570516231525e-06*delta_n*delta_t + 0.0011251273020889668*delta_n*v + 4.9114361114335354e-09*delta_n*delta_t**2 + 0.03279426174278255*delta_n*r**2 -5.946117028410193e-08*delta_t*u**2 + 1.1306310777796303e-05*delta_t*v**2 + 3.3190501269843805e-10*delta_t**2*u -0.0011467582186509059*u*v**2 + 9.018856112370991e-08*delta_t**2*v + 0.0012651422425927622*u**2*v + 0.015191554927764361*r**2*u**2 + 1.1375006351434954e-06*delta_n*delta_t*r -1.9872024426422902e-07*delta_n*delta_t*u -1.7916605582820277e-06*delta_n*delta_t*v -9.185627609026802e-05*delta_n*delta_t*r**2 -1.6644052762981315e-09*delta_n*delta_t**2*r -4.452349825126672e-05*delta_n*u*v + 0.0169578792393221*delta_n*r**2*u + 4.907055384612531e-06*delta_t*r*u**2 + 5.717806295840299e-05*delta_t*r**2*u -0.000813914658875696*delta_t*r**2*v -3.9646083548925386e-08*delta_t**2*r*u + 4.475393143190829e-09*delta_t**2*r*v + 0.12220372961142889*r**2*u*v -9.051680317981846e-06*2*delta_t*u*v -1.438935345690965e-06*delta_n*delta_t*r*u + 1.522701530279913e-05*delta_t*r*u*v'
	new_str_simple_split = ['-1.8790570516231525e-06*delta_n*delta_t ','+ 0.0011251273020889668*delta_n*v ','+ 4.9114361114335354e-09*delta_n*delta_t**2 ','+ 0.03279426174278255*delta_n*r**2 ','-5.946117028410193e-08*delta_t*u**2',' + 1.1306310777796303e-05*delta_t*v**2 ','+ 3.3190501269843805e-10*delta_t**2*u ','-0.0011467582186509059*u*v**2',' + 9.018856112370991e-08*delta_t**2*v ','+ 0.0012651422425927622*u**2*v',' + 0.015191554927764361*r**2*u**2 ','+ 1.1375006351434954e-06*delta_n*delta_t*r',' -1.9872024426422902e-07*delta_n*delta_t*u ','-1.7916605582820277e-06*delta_n*delta_t*v ','-9.185627609026802e-05*delta_n*delta_t*r**2 ','-1.6644052762981315e-09*delta_n*delta_t**2*r ','-4.452349825126672e-05*delta_n*u*v ','+ 0.0169578792393221*delta_n*r**2*u ','+ 4.907055384612531e-06*delta_t*r*u**2 ','+ 5.717806295840299e-05*delta_t*r**2*u',' -0.000813914658875696*delta_t*r**2*v ','-3.9646083548925386e-08*delta_t**2*r*u',' + 4.475393143190829e-09*delta_t**2*r*v ','+ 0.12220372961142889*r**2*u*v ','-9.051680317981846e-06*2*delta_t*u*v ','-1.438935345690965e-06*delta_n*delta_t*r*u ',' + 1.522701530279913e-05*delta_t*r*u*v']
	
	new_str_simple_reduced = '-1.8790570516231525e-06*delta_n*delta_t + 0.0011251273020889668*delta_n*v + 4.9114361114335354e-09*delta_n*delta_t**2 + 0.03279426174278255*delta_n*r**2 + 1.1306310777796303e-05*delta_t*v**2  -0.0011467582186509059*u*v**2 + 9.018856112370991e-08*delta_t**2*v + 0.0012651422425927622*u**2*v + 0.015191554927764361*r**2*u**2  -1.9872024426422902e-07*delta_n*delta_t*u -1.7916605582820277e-06*delta_n*delta_t*v -9.185627609026802e-05*delta_n*delta_t*r**2  + 0.0169578792393221*delta_n*r**2*u + 4.907055384612531e-06*delta_t*r*u**2 + 5.717806295840299e-05*delta_t*r**2*u -0.000813914658875696*delta_t*r**2*v -3.9646083548925386e-08*delta_t**2*r*u  + 0.12220372961142889*r**2*u*v -9.051680317981846e-06*2*delta_t*u*v -1.438935345690965e-06*delta_n*delta_t*r*u + 1.522701530279913e-05*delta_t*r*u*v'
	new_str_simple_reduced_split = ['-1.8790570516231525e-06*delta_n*delta_t ','+ 0.0011251273020889668*delta_n*v ','+ 4.9114361114335354e-09*delta_n*delta_t**2 ','+ 0.03279426174278255*delta_n*r**2 ','+ 1.1306310777796303e-05*delta_t*v**2 ',' -0.0011467582186509059*u*v**2 ','+ 9.018856112370991e-08*delta_t**2*v',' + 0.0012651422425927622*u**2*v',' + 0.015191554927764361*r**2*u**2 ',' -1.9872024426422902e-07*delta_n*delta_t*u ','-1.7916605582820277e-06*delta_n*delta_t*v ','-9.185627609026802e-05*delta_n*delta_t*r**2 ',' + 0.0169578792393221*delta_n*r**2*u ','+ 4.907055384612531e-06*delta_t*r*u**2 ','+ 5.717806295840299e-05*delta_t*r**2*u ','-0.000813914658875696*delta_t*r**2*v',' -3.9646083548925386e-08*delta_t**2*r*u  ','+ 0.12220372961142889*r**2*u*v ','-9.051680317981846e-06*2*delta_t*u*v ','-1.438935345690965e-06*delta_n*delta_t*r*u ','+ 1.522701530279913e-05*delta_t*r*u*v']
	

	## functions
	func_simple = str2func(str_simple)
	func_complex = str2func(str_complex)
	new_func = str2func(new_str)
	new_func_simple = str2func(new_str_simple)
	new_func_simple_reduced = str2func(new_str_simple_reduced)


	### arrays
	sol_simple = func_simple(u,v,r,delta_t,delta_n)
	sol_complex = func_complex(u,v,r,delta_t,delta_n)
	new_sol = new_func(u,v,r,delta_t,delta_n)
	new_sol_simple = new_func_simple(u,v,r,delta_t,delta_n)
	new_sol_simple_reduced = new_func_simple_reduced(u,v,r,delta_t,delta_n)
	sol = new_sol_simple_reduced.copy()
	### plot solution

	plt.figure()
	plt.plot(time, y)
	plt.plot(time, new_sol_simple)
	plt.plot(time, new_sol_simple_reduced)
	plt.grid()
	plt.legend(['GT','new model', 'new model reduced'])
	

	#### plot of a subsection
	#str
	square = '+ 4.9114361114335354e-09*delta_n*delta_t**2 '
	sinusodial = '+ 9.018856112370991e-08*delta_t**2*v'
	
	#func
	func_square = str2func(square)
	func_sin = str2func(sinusodial)

	#sol 
	sol_square = func_square(u,v,r,delta_t,delta_n)
	sol_sin = func_sin(u,v,r,delta_t,delta_n)

	plt.figure()
	plt.subplot(311)
	plt.plot(time, sol_square)
	plt.legend([square])
	plt.grid()
	plt.subplot(312)	
	plt.plot(time, sol_sin)
	plt.legend([sinusodial])
	plt.grid()
	plt.subplot(313)
	plt.plot(time, sol_sin + sol_square)
	plt.plot(time, y)
	plt.grid()
	plt.legend(['Sum of the above', 'Ground Truth'])
	#plt.show()


	## parts plot
	plot_parts(new_str_simple_reduced_split)
	plot_parts_w_sol(new_str_simple_reduced_split, y, xlim = [450, 610], ylim = [-0.12, 0.12])
	plot_parts_w_sol(new_str_simple_reduced_split, y, xlim = [2253, 2345], ylim = [-0.3, 0.3])
	plot_parts_w_sol(new_str_simple_reduced_split, y, xlim = [3080, 3160], ylim = [-0.5, 0.5])
	plot_parts_w_sol(new_str_simple_reduced_split, y, xlim = [3640, 3740], ylim = [-0.5, 0.7])
	plt.show()


# ###-- get data
# if 1:
# 	X1 = np.load('/home/gislehalv/Master/Data/numpy_data_from_bag/' + 'bag1_1'+'.npy')
# 	X2 = np.load('/home/gislehalv/Master/Data/numpy_data_from_bag/' + 'bag2_1'+'.npy')
# 	X3 = np.load('/home/gislehalv/Master/Data/numpy_data_from_bag/' + 'bag3_1'+'.npy')
# 	X4 = np.load('/home/gislehalv/Master/Data/numpy_data_from_bag/' + 'bag4_1'+'.npy')


# 	#fix time 
# 	X2[-1] = X2[-1] + X1[-1, -1]
# 	X3[-1] = X3[-1] + X2[-1, -1]
# 	#X4[-1] = X4[-1] + X3[-1, -1]

# 	#melt all the data together
# 	X = np.concatenate((X1,X2,X3),axis = 1)

# 	#remove data with bucket < 95
# 	index = []
# 	for i in range(np.shape(X)[1]):
# 		if np.shape(X)[1] > i:
# 			if X[-2, i] < 95:
# 				index.append(i)
# 	X = np.delete(X, index, 1)


# 	u = X[0]
# 	v = X[1]
# 	r = X[2]
# 	delta_t = X[-4]
# 	delta_n = X[-3]
# 	y = X[4]        #  <----dr btw
# 	time = X[-1]
# 	delta_n[delta_n > 27] = 27 #remove error in the data
# 	delta_n[delta_n < -27] = -27
# #### --------- dv ------------ 
# if dv_:
# 	### str
# 	orig_str = '-3.952212504979791e-06*delta_n**3*u -4.323806251907643e-06*delta_n**3*v + 3.633763321792768e-06*delta_n**2*u**2 -1.831870024171225e-07*delta_n*delta_t*u**2 + 0.0003823435101753467*delta_n*u**2*sin(cos(r)) -2.1912674505216164e-09*delta_n**2*delta_t*u -8.612870897527819e-07*delta_n**2*delta_t*v -3.3142056104988395e-09*delta_t*u**2*sin(cos(r)) + 7.719118258445558e-05*delta_n**2*u*v + 5.402037395386755e-06*delta_n**2*u*sin(cos(r)) + 0.0005809985939579979*delta_n**2*v*sin(cos(r)) + 6.585244046079131e-07*delta_n*delta_t*u*v -7.821381053574049e-07*delta_n*delta_t*u*sin(cos(r)) + 6.313586874164905e-06*delta_n*delta_t*v*sin(cos(r)) -0.0023077638335907077*delta_n*u*v*sin(cos(r)) -1.4485249323632313e-05*delta_t*u*v*sin(cos(r))'
# 	orig_str_split = ['-3.952212504979791e-06*delta_n**3*u ','-4.323806251907643e-06*delta_n**3*v ','+ 3.633763321792768e-06*delta_n**2*u**2 ','-1.831870024171225e-07*delta_n*delta_t*u**2 ','+ 0.0003823435101753467*delta_n*u**2*sin(cos(r))',' -2.1912674505216164e-09*delta_n**2*delta_t*u',' -8.612870897527819e-07*delta_n**2*delta_t*v',' -3.3142056104988395e-09*delta_t*u**2*sin(cos(r))',' + 7.719118258445558e-05*delta_n**2*u*v',' + 5.402037395386755e-06*delta_n**2*u*sin(cos(r))',' + 0.0005809985939579979*delta_n**2*v*sin(cos(r)) ','+ 6.585244046079131e-07*delta_n*delta_t*u*v ','-7.821381053574049e-07*delta_n*delta_t*u*sin(cos(r)) ','+ 6.313586874164905e-06*delta_n*delta_t*v*sin(cos(r)) ','-0.0023077638335907077*delta_n*u*v*sin(cos(r)) ',' -1.4485249323632313e-05*delta_t*u*v*sin(cos(r))']
# 	new_str = '-1.7834290681198802e-07*delta_t**2*v + 6.706306888210264e-10*delta_t**2*Abs(u) + 1.076964429624133e-06*delta_n*delta_t*v -1.5462579978451662e-06*delta_n*delta_t*Abs(u) + 0.008807113767162776*delta_n*r*v + 0.0061839222365966135*delta_n*r*Abs(u) + 0.0002921184004195739*delta_t*r*v -9.297713363841411e-05*delta_t*r*Abs(u)'


# 	str_no_low_vals = '-3.952212504979791e-06*delta_n**3*u -4.323806251907643e-06*delta_n**3*v + 3.633763321792768e-06*delta_n**2*u**2 -1.831870024171225e-07*delta_n*delta_t*u**2 + 0.0003823435101753467*delta_n*u**2*sin(cos(r)) -8.612870897527819e-07*delta_n**2*delta_t*v + 7.719118258445558e-05*delta_n**2*u*v  + 0.0005809985939579979*delta_n**2*v*sin(cos(r)) + 6.585244046079131e-07*delta_n*delta_t*u*v -7.821381053574049e-07*delta_n*delta_t*u*sin(cos(r)) + 6.313586874164905e-06*delta_n*delta_t*v*sin(cos(r)) -0.0023077638335907077*delta_n*u*v*sin(cos(r)) -1.4485249323632313e-05*delta_t*u*v*sin(cos(r))'
# 	str_no_sincos = '-3.952212504979791e-06*delta_n**3*u -4.323806251907643e-06*delta_n**3*v + 3.633763321792768e-06*delta_n**2*u**2 -1.831870024171225e-07*delta_n*delta_t*u**2 + 0.0003823435101753467*delta_n*u**2*0.82 -8.612870897527819e-07*delta_n**2*delta_t*v + 7.719118258445558e-05*delta_n**2*u*v  + 0.0005809985939579979*delta_n**2*v*0.82 + 6.585244046079131e-07*delta_n*delta_t*u*v -7.821381053574049e-07*delta_n*delta_t*u*0.82 + 6.313586874164905e-06*delta_n*delta_t*v*0.82 -0.0023077638335907077*delta_n*u*v*0.82 -1.4485249323632313e-05*delta_t*u*v*0.82'
# 	str_no_sincos_split = ['-3.952212504979791e-06*delta_n**3*u',' -4.323806251907643e-06*delta_n**3*v',' + 3.633763321792768e-06*delta_n**2*u**2 ','-1.831870024171225e-07*delta_n*delta_t*u**2 ','+ 0.0003823435101753467*delta_n*u**2*0.82 ','-8.612870897527819e-07*delta_n**2*delta_t*v ','+ 7.719118258445558e-05*delta_n**2*u*v  ','+ 0.0005809985939579979*delta_n**2*v*0.82 ','+ 6.585244046079131e-07*delta_n*delta_t*u*v ','-7.821381053574049e-07*delta_n*delta_t*u*0.82 ','+ 6.313586874164905e-06*delta_n*delta_t*v*0.82',' -0.0023077638335907077*delta_n*u*v*0.82 ','-1.4485249323632313e-05*delta_t*u*v*0.82']
	
# 	str_remove_stuff = '-3.952212504979791e-06*delta_n**3*u -4.323806251907643e-06*delta_n**3*v  + 0.0003823435101753467*delta_n*u**2*sin(cos(r)) -8.612870897527819e-07*delta_n**2*delta_t*v + 7.719118258445558e-05*delta_n**2*u*v  + 0.0005809985939579979*delta_n**2*v*sin(cos(r)) + 6.585244046079131e-07*delta_n*delta_t*u*v -7.821381053574049e-07*delta_n*delta_t*u*sin(cos(r)) + 6.313586874164905e-06*delta_n*delta_t*v*sin(cos(r)) -0.0023077638335907077*delta_n*u*v*sin(cos(r)) -1.4485249323632313e-05*delta_t*u*v*sin(cos(r))'

# 	##funcs
# 	new_func = str2func(new_str)
# 	orig_func = str2func(orig_str)
# 	func_no_low = str2func(str_no_low_vals)
# 	func_no_sincos = str2func(str_no_sincos)
# 	func_remove_stuff = str2func(str_remove_stuff)
	


# 	#arrays 
# 	sol = orig_func(u,v,r,delta_t,delta_n)
# 	sol_no_low = func_no_low(u,v,r,delta_t,delta_n)
# 	sol_no_sincos = func_no_sincos(u,v,r,delta_t,delta_n)
# 	sol_remove_suff = func_remove_stuff(u,v,r,delta_t,delta_n)
# 	new_sol = new_func(u,v,r,delta_t,delta_n)


# 	#error
# 	# plt.figure()
# 	# plt.plot(time, y - sol)
# 	# plt.grid()
# 	# plt.xlabel('Time [s]')
# 	# plt.ylabel('error')
 
# 	#plot no low vals - EQUAL
# 	# plt.figure()
# 	# plt.plot(time, sol)
# 	# plt.plot(time, sol_no_low)
# 	# plt.plot(time, y)
# 	# plt.legend(['orig', 'no low', 'GT'])
# 	# plt.show()

# 	# plot no sin(cos(r))
# 	# plt.figure()
# 	# plt.plot(time, sol)
# 	# plt.plot(time, sol_no_sincos)
# 	# plt.plot(time, y)
# 	# plt.grid()
# 	# plt.legend(['orig', 'no sin(cos(r))', 'GT'])
# 	# plt.show()

# 	#plot - removed delta_n^2*u^2 and delta_n*delta_t*u^2 - looks shit
# 	# plt.figure()
# 	# plt.plot(time, sol)
# 	# plt.plot(time, sol_remove_suff)
# 	# plt.plot(time, y)
# 	# plt.grid()
# 	# plt.legend(['orig', 'removed stuff', 'GT'])
# 	# plt.show()


# 	#plot solution
# 	plt.figure()
# 	plt.plot(time, sol)
# 	plt.plot(time, y)
# 	plt.plot(time, new_sol)
# 	plt.grid()
# 	plt.legend(['old sol', 'gt', ' new sol'])
# 	#plt.show()

# 	# #plot_parts(str_no_sincos_split)
# 	# new_param = least_sq(str_no_sincos_split)


# 	# plt.plot(time, new_param)

# 	# plt.legend(['Model', 'Ground Truth', 'New params from LS'])
# 	# plt.xlabel('Time [s]')
# 	# plt.ylabel('dr [m/s^2]')
# 	# plt.show()


# ###-- get data
# if 1:
# 	X1 = np.load('/home/gislehalv/Master/Data/numpy_data_from_bag/' + 'bag1_3'+'.npy')
# 	X2 = np.load('/home/gislehalv/Master/Data/numpy_data_from_bag/' + 'bag2_3'+'.npy')
# 	X3 = np.load('/home/gislehalv/Master/Data/numpy_data_from_bag/' + 'bag3_3'+'.npy')
# 	X4 = np.load('/home/gislehalv/Master/Data/numpy_data_from_bag/' + 'bag4_3'+'.npy')


# 	#fix time 
# 	X2[-1] = X2[-1] + X1[-1, -1]
# 	X3[-1] = X3[-1] + X2[-1, -1]
# 	#X4[-1] = X4[-1] + X3[-1, -1]

# 	#melt all the data together
# 	X = np.concatenate((X1,X2,X3),axis = 1)

# 	#remove data with bucket < 95
# 	index = []
# 	for i in range(np.shape(X)[1]):
# 		if np.shape(X)[1] > i:
# 			if X[-2, i] < 95:
# 				index.append(i)
# 	X = np.delete(X, index, 1)


# 	u = X[0]
# 	v = X[1]
# 	r = X[2]
# 	delta_t = X[-4]
# 	delta_n = X[-3]
# 	y = X[4]        #  <----dr btw
# 	time = X[-1]
# 	delta_n[delta_n > 27] = 27 #remove error in the data
# 	delta_n[delta_n < -27] = -27


# #### --------- dv ------------ 
# if dv_:
# 	### str
# 	orig_str = '-3.952212504979791e-06*delta_n**3*u -4.323806251907643e-06*delta_n**3*v + 3.633763321792768e-06*delta_n**2*u**2 -1.831870024171225e-07*delta_n*delta_t*u**2 + 0.0003823435101753467*delta_n*u**2*sin(cos(r)) -2.1912674505216164e-09*delta_n**2*delta_t*u -8.612870897527819e-07*delta_n**2*delta_t*v -3.3142056104988395e-09*delta_t*u**2*sin(cos(r)) + 7.719118258445558e-05*delta_n**2*u*v + 5.402037395386755e-06*delta_n**2*u*sin(cos(r)) + 0.0005809985939579979*delta_n**2*v*sin(cos(r)) + 6.585244046079131e-07*delta_n*delta_t*u*v -7.821381053574049e-07*delta_n*delta_t*u*sin(cos(r)) + 6.313586874164905e-06*delta_n*delta_t*v*sin(cos(r)) -0.0023077638335907077*delta_n*u*v*sin(cos(r)) -1.4485249323632313e-05*delta_t*u*v*sin(cos(r))'
# 	orig_str_split = ['-3.952212504979791e-06*delta_n**3*u ','-4.323806251907643e-06*delta_n**3*v ','+ 3.633763321792768e-06*delta_n**2*u**2 ','-1.831870024171225e-07*delta_n*delta_t*u**2 ','+ 0.0003823435101753467*delta_n*u**2*sin(cos(r))',' -2.1912674505216164e-09*delta_n**2*delta_t*u',' -8.612870897527819e-07*delta_n**2*delta_t*v',' -3.3142056104988395e-09*delta_t*u**2*sin(cos(r))',' + 7.719118258445558e-05*delta_n**2*u*v',' + 5.402037395386755e-06*delta_n**2*u*sin(cos(r))',' + 0.0005809985939579979*delta_n**2*v*sin(cos(r)) ','+ 6.585244046079131e-07*delta_n*delta_t*u*v ','-7.821381053574049e-07*delta_n*delta_t*u*sin(cos(r)) ','+ 6.313586874164905e-06*delta_n*delta_t*v*sin(cos(r)) ','-0.0023077638335907077*delta_n*u*v*sin(cos(r)) ',' -1.4485249323632313e-05*delta_t*u*v*sin(cos(r))']
# 	new_str = '-1.7834290681198802e-07*delta_t**2*v + 6.706306888210264e-10*delta_t**2*Abs(u) + 1.076964429624133e-06*delta_n*delta_t*v -1.5462579978451662e-06*delta_n*delta_t*Abs(u) + 0.008807113767162776*delta_n*r*v + 0.0061839222365966135*delta_n*r*Abs(u) + 0.0002921184004195739*delta_t*r*v -9.297713363841411e-05*delta_t*r*Abs(u)'


# 	str_no_low_vals = '-3.952212504979791e-06*delta_n**3*u -4.323806251907643e-06*delta_n**3*v + 3.633763321792768e-06*delta_n**2*u**2 -1.831870024171225e-07*delta_n*delta_t*u**2 + 0.0003823435101753467*delta_n*u**2*sin(cos(r)) -8.612870897527819e-07*delta_n**2*delta_t*v + 7.719118258445558e-05*delta_n**2*u*v  + 0.0005809985939579979*delta_n**2*v*sin(cos(r)) + 6.585244046079131e-07*delta_n*delta_t*u*v -7.821381053574049e-07*delta_n*delta_t*u*sin(cos(r)) + 6.313586874164905e-06*delta_n*delta_t*v*sin(cos(r)) -0.0023077638335907077*delta_n*u*v*sin(cos(r)) -1.4485249323632313e-05*delta_t*u*v*sin(cos(r))'
# 	str_no_sincos = '-3.952212504979791e-06*delta_n**3*u -4.323806251907643e-06*delta_n**3*v + 3.633763321792768e-06*delta_n**2*u**2 -1.831870024171225e-07*delta_n*delta_t*u**2 + 0.0003823435101753467*delta_n*u**2*0.82 -8.612870897527819e-07*delta_n**2*delta_t*v + 7.719118258445558e-05*delta_n**2*u*v  + 0.0005809985939579979*delta_n**2*v*0.82 + 6.585244046079131e-07*delta_n*delta_t*u*v -7.821381053574049e-07*delta_n*delta_t*u*0.82 + 6.313586874164905e-06*delta_n*delta_t*v*0.82 -0.0023077638335907077*delta_n*u*v*0.82 -1.4485249323632313e-05*delta_t*u*v*0.82'
# 	str_no_sincos_split = ['-3.952212504979791e-06*delta_n**3*u',' -4.323806251907643e-06*delta_n**3*v',' + 3.633763321792768e-06*delta_n**2*u**2 ','-1.831870024171225e-07*delta_n*delta_t*u**2 ','+ 0.0003823435101753467*delta_n*u**2*0.82 ','-8.612870897527819e-07*delta_n**2*delta_t*v ','+ 7.719118258445558e-05*delta_n**2*u*v  ','+ 0.0005809985939579979*delta_n**2*v*0.82 ','+ 6.585244046079131e-07*delta_n*delta_t*u*v ','-7.821381053574049e-07*delta_n*delta_t*u*0.82 ','+ 6.313586874164905e-06*delta_n*delta_t*v*0.82',' -0.0023077638335907077*delta_n*u*v*0.82 ','-1.4485249323632313e-05*delta_t*u*v*0.82']
	
# 	str_remove_stuff = '-3.952212504979791e-06*delta_n**3*u -4.323806251907643e-06*delta_n**3*v  + 0.0003823435101753467*delta_n*u**2*sin(cos(r)) -8.612870897527819e-07*delta_n**2*delta_t*v + 7.719118258445558e-05*delta_n**2*u*v  + 0.0005809985939579979*delta_n**2*v*sin(cos(r)) + 6.585244046079131e-07*delta_n*delta_t*u*v -7.821381053574049e-07*delta_n*delta_t*u*sin(cos(r)) + 6.313586874164905e-06*delta_n*delta_t*v*sin(cos(r)) -0.0023077638335907077*delta_n*u*v*sin(cos(r)) -1.4485249323632313e-05*delta_t*u*v*sin(cos(r))'

# 	##funcs
# 	new_func = str2func(new_str)
# 	orig_func = str2func(orig_str)
# 	func_no_low = str2func(str_no_low_vals)
# 	func_no_sincos = str2func(str_no_sincos)
# 	func_remove_stuff = str2func(str_remove_stuff)
	


# 	#arrays 
# 	sol = orig_func(u,v,r,delta_t,delta_n)
# 	sol_no_low = func_no_low(u,v,r,delta_t,delta_n)
# 	sol_no_sincos = func_no_sincos(u,v,r,delta_t,delta_n)
# 	sol_remove_suff = func_remove_stuff(u,v,r,delta_t,delta_n)
# 	new_sol = new_func(u,v,r,delta_t,delta_n)


# 	#error
# 	# plt.figure()
# 	# plt.plot(time, y - sol)
# 	# plt.grid()
# 	# plt.xlabel('Time [s]')
# 	# plt.ylabel('error')
 
# 	#plot no low vals - EQUAL
# 	# plt.figure()
# 	# plt.plot(time, sol)
# 	# plt.plot(time, sol_no_low)
# 	# plt.plot(time, y)
# 	# plt.legend(['orig', 'no low', 'GT'])
# 	# plt.show()

# 	# plot no sin(cos(r))
# 	# plt.figure()
# 	# plt.plot(time, sol)
# 	# plt.plot(time, sol_no_sincos)
# 	# plt.plot(time, y)
# 	# plt.grid()
# 	# plt.legend(['orig', 'no sin(cos(r))', 'GT'])
# 	# plt.show()

# 	#plot - removed delta_n^2*u^2 and delta_n*delta_t*u^2 - looks shit
# 	# plt.figure()
# 	# plt.plot(time, sol)
# 	# plt.plot(time, sol_remove_suff)
# 	# plt.plot(time, y)
# 	# plt.grid()
# 	# plt.legend(['orig', 'removed stuff', 'GT'])
# 	# plt.show()


# 	#plot solution
# 	plt.figure()
# 	plt.plot(time, sol)
# 	plt.plot(time, y)
# 	plt.plot(time, new_sol)
# 	plt.grid()
# 	plt.legend(['old sol', 'gt', ' new sol'])
# 	plt.show()

# 	# #plot_parts(str_no_sincos_split)
# 	# new_param = least_sq(str_no_sincos_split)


# 	# plt.plot(time, new_param)

# 	# plt.legend(['Model', 'Ground Truth', 'New params from LS'])
# 	# plt.xlabel('Time [s]')
# 	# plt.ylabel('dr [m/s^2]')
# 	# plt.show()








# #### new data
# bag_1 = 'hal_control_2018-12-11-10-53-26_0' #large!
# bag_2 = 'hal_control_2018-12-11-11-49-22_0' #similar to bag1 but smaller
# bag_3 = 'hal_control_2018-12-11-12-13-58_0' #
# bag_4 = 'hal_control_2018-12-11-12-13-58_0'

# # bag path
# path = '/home/gislehalv/Master/Data/'

# bagFile1 = path + bag_1 + '.bag'
# bagFile2 = path + bag_2 + '.bag'
# bagFile3 = path + bag_3 + '.bag'
# bagFile4 = path + bag_4 + '.bag'

# # get data
# X1 = my_lib.open_bag(bagFile1, plot=False, thr_bucket = False, filter_cutoff = 0.3)
# X2 = my_lib.open_bag(bagFile2, plot=False, thr_bucket = False, filter_cutoff = 0.3)
# X3 = my_lib.open_bag(bagFile3, plot=False, thr_bucket = False, filter_cutoff = 0.3)
# X4 = my_lib.open_bag(bagFile4, plot=False, thr_bucket = False, filter_cutoff = 0.3)



# path_to_dir = '/home/gislehalv/Master/Data/numpy_data_from_bag/'
# np.save(path_to_dir + 'bag1_3', X1)
# np.save(path_to_dir + 'bag2_3', X2)
# np.save(path_to_dir + 'bag3_3', X3)
# np.save(path_to_dir + 'bag4_3', X4)

# exit()




# #fix time
# X2[-1] = X2[-1] + X1[-1, -1]
# X3[-1] = X3[-1] + X2[-1, -1]
# X4[-1] = X4[-1] + X3[-1, -1]


# X = np.concatenate((X1,X2,X3,X4),axis = 1)


# #remove data with bucket < 95
# index = []
# for i in range(np.shape(X)[1]):
# 	if np.shape(X)[1] > i:
# 		if X[-2, i] < 95:
# 			index.append(i)
# X = np.delete(X, index, 1)

# u = X[0]
# v = X[1]
# r = X[2]
# delta_t = X[-4]
# delta_n = X[-3]
# y = X[5]        #  <----dr btw
# time = X[-1]
# delta_n[delta_n > 27] = 27 #remove error in the data
# delta_n[delta_n < -27] = -27



# sol_simple = func_simple(u,v,r,delta_t,delta_n)
# sol_complex = func_complex(u,v,r,delta_t,delta_n)


# ### plot solution

# plt.figure()
# plt.plot(time, y)
# plt.plot(time, sol_simple)
# plt.plot(time, sol_complex)
# plt.grid()
# plt.legend(['GT','model simple', 'model complex'])

# plt.figure()
# plt.subplot(211)
# plt.plot(time, delta_t)
# plt.grid()
# plt.subplot(212)
# plt.plot(time, delta_n)
# plt.grid()
# plt.show()











# def least_sq(func_list):
# 	F_list  = []
# 	for func in func_list:
# 		F_list.append(str2func(func))
# 	F = np.zeros((len(y), len(F_list)))

# 	for i, function in enumerate(F_list):
# 		F[:,i] = np.squeeze(function(u, v, r, delta_t, delta_n))

# 	F_trans = np.transpose(F)
# 	try:
# 		p = np.dot(np.linalg.inv(np.dot(F_trans,F)),np.dot(F_trans,y))  
# 	except:
# 		print('Singular Matrix for: ', individual)
# 		exit()
# 		mse = 1000 # large number
# 		return(mse,)

# 	tot_func = np.zeros((len(y)))

# 	for i, func in enumerate(F_list):
# 		tot_func = np.add(tot_func, p[i]*func(u, v, r, delta_t, delta_n))

# 	return tot_func

# tot_func = least_sq(func_list)

# plt.figure()
# plt.plot(time, sol)
# plt.plot(time, tot_func)
# plt.legend(['gt', 'pred'])
# plt.show()
# exit()

# #performance plot
# if 0:
# 	plt.figure()
# 	plt.plot(time, sol)
# 	plt.plot(time, y)
# 	plt.ylabel('du/dt')
# 	plt.xlabel('Time [s]')
# 	plt.legend(['Model', 'measured du'])
# 	plt.grid()

# 	#error plot
# 	plt.figure()
# 	plt.plot(time, np.subtract(sol,y))
# 	plt.ylabel('Error')
# 	plt.xlabel('Time [s]')
# 	plt.grid()
# 	plt.show()

# 	exit()

#mse = math.fsum((y - sol)**2)/len(sol)


####---- Find the parts of the eq that have tooo small values

#removed: - 5.0753014272873145*sin(r**2) - looks good on bag 4 and 3, not bag 1 and 2 -> placed back


#  
#changed 
