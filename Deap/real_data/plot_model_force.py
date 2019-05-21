"""

Plot solutions
"""

from sympy import sympify, cos, sin, expand, collect, Lambda, lambdify, symbols
import numpy as np
import matplotlib.pyplot as plt


##takes in string -> gives out the function for the string
def str2func(exp_str):
	u, v, r, fx, fy, fz = symbols('u v r fx, fy, fz')
	f = lambdify((u,v,r, fx, fy, fz), exp_str, 'numpy')
	return f


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
	

def load():
	### load data 
	path = '/home/gislehalv/Master/Data/numpy_data_from_bag_force/'
	file = 'all_bags_cut1.npy'
	X = np.load(path+file)


	#remove data with bucket < 95
	index = []
	for i in range(np.shape(X)[1]):
		if np.shape(X)[1] > i:
			if X[-2, i] < 95:
				index.append(i)
	X = np.delete(X, index, 1)

	return X



X = load()

# divide into val and train sets
X_test = X[:, list(range(100000, len(X[0])))]


X_val = np.concatenate((X[:,11511:18500], X[:,27912:32693], X[:,56531:60714], X[:,70475:74193]),axis = 1)
index = list(range(11511,18500)) + list(range(27912,32693)) +list(range(56531,60714)) + list(range(70475,74193)) + list(range(100000, np.shape(X)[1]))

X = np.delete(X, index, 1)



du_orig_str = ' 0.000264871203432643*fx -5.104448463432132e-07*fx*u**3 -2.773742056836302e-06*Abs(fz*v) + 2.649152675182239e-09*fx*Abs(fx) -0.00010529881090182071*r*v*Abs(fx) -0.037022654378220055*u**2 + 0.0020864289417205986*u**3 -1.722147024416803e-05*fx*r -9.437055733162733e-05*fx*u + 1.348464006791731e-05*fx*u**2 + 0.04778449754855396*r*u + 2.2407149473159294*r*v + 0.00947558535982701*u*v -0.0007198331382015782*u**2*v'

du_func = str2func(du_orig_str)
du_array = du_func(X[0], X[1], X[2], X[6], X[7], X[8])






######### plots

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

plt.figure()
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), X[3])
plt.plot(np.arange(0, len(X[-1])*0.05, 0.05), du_array)
plt.xlabel('Time [s]')
plt.ylabel('$\dot{u} $')
plt.legend('')
plt.grid()

plt.show()