"""
plotting to analyze the 3 models for the 3 pahses  
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

### what pahse and eq to use

dis = False
semi_dis = True
planing = False

du = True
dv = False
dr = False


### import data ###
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

	lpp = 10.5
	g = 9.81

	index_dis = []
	index_semi_dis = []
	index_planing = []



	for i in range(np.shape(X)[1]):
		U = np.sqrt(X[0, i]**2 + X[1, i]**2) 
		Froude = U/np.sqrt(g*lpp)

		# displacement phase
		if Froude < 0.4:
			index_dis.append(i)

		# semi-dis phase 
		elif Froude <= 1 and Froude >= 0.4:
			index_semi_dis.append(i)

		#planing phase
		elif Froude > 1:
			index_planing.append(i)
	X_dis = np.take(X,index_dis,axis = 1)
	X_semi_dis = np.take(X, index_semi_dis, axis = 1)
	X_planing = np.take(X, index_planing, axis = 1)

	if dis:
		X = X_dis.copy()
	elif semi_dis:
		X = X_semi_dis.copy()
	elif planing:
		X = X_planing.copy() 

	if du:
		y = X[3]
	elif dv:
		y = X[4]
	elif dr:
		y = X[5]




	u = X[0]
	v = X[1]
	r = X[2]
	delta_t = X[-4]
	delta_n = X[-3]
	time = X[-1]
	delta_n[delta_n > 27] = 27 #remove error in the data
	delta_n[delta_n < -27] = -27
		

#######  du - dissplacement phase

if du:
	if dis:
		orig_str = '+ 4.975146987585405e-08*delta_t**2 + 1.1198147168508188e-09*delta_t**3 -0.04065430599442876*u**3 + 5.787416954839038e-05*delta_t*u + 6.484074411385043e-05*delta_t*v -0.009368401112740399*delta_t*r**2 + 5.161352837908793e-06*delta_t**2*r + 2.582784694777203*r**2*u + 1.0536370823003376e-06*delta_t**2*v -0.7399222220566095*r**2*v + 0.1260266084851711*u**2*v + 0.00013826432655201517*2*delta_t*u**2 + 0.3080270859431593*2*r*u**2 -4.318855250214485e-07*2*delta_t**2*u + 0.0007552873748542799*delta_t*r*v -0.0007368852104889176*delta_t*u*v + 0.09500762817020458*2*r*u*v -0.0011765537987994679*3*delta_t*r*u'
		
		orig_func = str2func(orig_str)

		orig_array = orig_func(u, v, r, delta_t, delta_n)

		plt.figure()
		plt.plot(time, orig_array)
		plt.plot(time, y)
		plt.grid()
		plt.legend(['predictions', 'gt'])
		plt.show()

	if semi_dis:
		orig_str = '+ 7.721218673685614e-07*delta_t**2 + 0.026061892210278326*u**2 -0.0008535460603761452*u**3 -7.29295128628878e-07*delta_t*u**2 + 0.00559624266176921*r*u**2 -0.0035325479529925152*u**2*v -2.2564008305253277e-05*2*delta_t*u -2.881914803154245e-05*delta_t*r*u + 0.002690650969450826*delta_t*r*v + 1.4247063948045861e-05*delta_t*u*v -0.3983364953976801*r*u*v -0.15615117151985536*sin(cos(u)) '
		orig_str = '-0.0006179593840363257*delta_t -0.09095280978444409*u + 6.63389216147781e-07*delta_t**2 -8.933632575028257e-05*delta_t*Abs(v) -4.226949786283376*r**2'
		orig_str = '-0.00040932018041908735*delta_t + 6.087888358026739e-07*delta_t**2 -0.0011211762510408903*delta_t*Abs(r) -0.11230006092248856*Abs(u)'
		
		orig_func = str2func(orig_str)

		orig_array = orig_func(u, v, r, delta_t, delta_n)

		plt.figure()
		plt.plot(list(range(len(time))), orig_array)
		plt.plot(list(range(len(time))), y)
		plt.grid()
		plt.legend(['predictions', 'gt'])
		plt.show()