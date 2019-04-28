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


### -- expand eq 
# eq = ' 3.8871966999318944e-5*delta_t*sin(v) + 0.04301652489771968*r*u + 1.5173766642286524e-6*u**2*(delta_t + v) - 0.1529942584024159*u - 5.0753014272873145*sin(r**2) + 0.20253043751653854*sin((sin(Abs(sin(r + u))) + Abs(u))*cos(cos(v))) - 0.11026599592186592*sin(cos(u)) - 0.23837246536136725*cos(r) + 0.005602719442464377*cos(Abs(u)) + 0.0005172216911381033*Abs(delta_t) + 3.1708546137642567e-10*Abs(delta_t**3) - 9.920301845328634e-12*Abs(delta_t**3*u) + 0.07847803745057924*Abs(sin(u) + cos(v)) + 0.015559091743164777*Abs(cos(r + v + sin(v)) + cos(sin(cos(delta_t) + cos(v)))) - 0.0003579585125443874*Abs(delta_n + delta_t + r**2)'
# expand_str = expand(eq)
# print(expand_str)
# exit()

###-----

#exp_str = '1.5173766642286524e-6*delta_t*u**2 + 3.8871966999318944e-5*delta_t*sin(v) + 0.04301652489771968*r*u + 1.5173766642286524e-6*u**2*v - 0.1529942584024159*u - 5.0753014272873145*sin(r**2) + 0.20253043751653854*sin(sin(Abs(sin(r + u)))*cos(cos(v)) + cos(cos(v))*Abs(u)) - 0.11026599592186592*sin(cos(u)) - 0.23837246536136725*cos(r) + 0.005602719442464377*cos(Abs(u)) + 0.0005172216911381033*Abs(delta_t) + 3.1708546137642567e-10*Abs(delta_t**3) - 9.920301845328634e-12*Abs(delta_t**3*u) + 0.07847803745057924*Abs(sin(u) + cos(v)) + 0.015559091743164777*Abs(cos(r + v + sin(v)) + cos(sin(cos(delta_t) + cos(v)))) - 0.0003579585125443874*Abs(delta_n + delta_t + r**2)'


#dr:
exp_str =  '2.2982880646793903e-06*delta_n*delta_t + 3.4204391778078214e-05*delta_n*u + 8.898612698155764e-05*delta_t*v -0.00887592642227475*u*v'


##takes in string -> gives out the function for the string
def str2func(exp_str):
	u, v, r, delta_t, delta_n = symbols('u v r delta_t delta_n')
	f = lambdify((u,v,r, delta_t, delta_n), exp_str, 'numpy')
	return f
func = str2func(exp_str)


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
delta_t = X[-4]
delta_n = X[-3]
y = X[5]        #  <----dr btw
time = X[-1]
delta_n[delta_n > 27] = 27 #remove error in the data
delta_n[delta_n < -27] = -27

sol = func(u,v,r,delta_t,delta_n)



plt.figure()
plt.plot( delta_t)
plt.plot(delta_n)
plt.grid()

plt.figure()
plt.plot(time,u)
plt.grid()
exit()
#### se whats the best parameters on the data with LS
#func_list is the functions without parameters
#func_list = ['Abs(sin(u) + cos(v))','Abs(delta_n + delta_t + r**2)','Abs(cos(r + v + sin(v)) + cos(sin(cos(delta_t) + cos(v)))) ','sin(r**2)','delta_t*u**2 ',' delta_t*sin(v) ','r*u ','sin(sin(Abs(sin(r + u)))*cos(cos(v)) + cos(cos(v))*Abs(u))','u**2*v ','u','sin(cos(u)) ','cos(r) ','cos(Abs(u)) ','Abs(delta_t) ','Abs(delta_t**3) ','Abs(delta_t**3*u)']

func_list =  ['delta_n*delta_t' , 'delta_n*u' , 'delta_t*v' , 'u*v']





F_list  = []
def least_sq(func_list):
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

tot_func = least_sq(func_list)

plt.figure()
plt.plot(time, sol)
plt.plot(time, tot_func)
plt.legend(['gt', 'pred'])
plt.show()
exit()

# #performance plot
if 0:
	plt.figure()
	plt.plot(time, sol)
	plt.plot(time, y)
	plt.ylabel('du/dt')
	plt.xlabel('Time [s]')
	plt.legend(['Model', 'measured du'])
	plt.grid()

	#error plot
	plt.figure()
	plt.plot(time, np.subtract(sol,y))
	plt.ylabel('Error')
	plt.xlabel('Time [s]')
	plt.grid()
	plt.show()

	exit()

#mse = math.fsum((y - sol)**2)/len(sol)


####---- Find the parts of the eq that have tooo small values

#removed: - 5.0753014272873145*sin(r**2) - looks good on bag 4 and 3, not bag 1 and 2 -> placed back


#  
#changed 

sub_func_list_small = ['+ 0.07847803745057924*Abs(sin(u) + cos(v))','- 0.0003579585125443874*Abs(delta_n + delta_t + r**2)','+ 0.015559091743164777*Abs(cos(r + v + sin(v)) + cos(sin(cos(delta_t) + cos(v)))) ','- 5.0753014272873145*sin(r**2)','1.5173766642286524e-6*delta_t*u**2 ',' + 3.8871966999318944e-5*delta_t*sin(v) ','+ 0.04301652489771968*r*u ','0.20253043751653854*sin(sin(Abs(sin(r + u)))*cos(cos(v)) + cos(cos(v))*Abs(u))','+ 1.5173766642286524e-6*u**2*v ','- 0.1529942584024159*u ','- 0.11026599592186592*sin(cos(u)) ','- 0.23837246536136725*cos(r) ','+ 0.005602719442464377*cos(Abs(u)) ','+ 0.0005172216911381033*Abs(delta_t) ','+ 3.1708546137642567e-10*Abs(delta_t**3) ','- 9.920301845328634e-12*Abs(delta_t**3*u) ',]
def remove_insig_parts(eq_list):
	func_list_rest  = []
	for sub_f in sub_func_list_small:
		sub_function = str2func(sub_f)
		if np.amax(np.abs(sub_function(u,v,r,delta_t,delta_n))) < 1e-2: 
			print('removed:', sub_f,'  the largest absolute value were:', np.amax(np.abs(sub_function(u,v,r,delta_t,delta_n))))
		else:
			func_list_rest.append(sub_f)
	return func_list_rest

#rest_eq_lits = remove_insig_parts(sub_func_list_small)
rest_eq_lits = sub_func_list_small

#plot evry part sep.
def plot_parts(func_list_rest):
	plt.figure()
	for i, sub_func in enumerate(func_list_rest):
		sub_function = str2func(sub_func)
		plt.subplot(len(func_list_rest), 1, i+1)
		new_func = sub_function(u,v,r,delta_t,delta_n)
		plt.plot(time, new_func)
		plt.legend([sub_func])

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
	plt.show()


plot_parts(rest_eq_lits)