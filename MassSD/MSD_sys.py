#Mass Spring Damper System

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumtrapz
import sys
import pydotplus
from gplearn.genetic import SymbolicRegressor
from scipy import signal
from sympy import sympify

sys.path.insert(0, '/home/gislehalv/Master/scripts')
import my_lib



#--------SIMULATION----
t,ddx,dx,x,inp = my_lib.MSD(time = [0,70,0.01], mdk = [2, 1, 1], tau = 'square')


#plot
plot = False
if plot:
	plt.figure()
	plt.subplot(312)
	plt.plot(t, dx)
	plt.grid()
	plt.ylabel('dx')
	plt.subplot(313)
	plt.plot(t,x)
	plt.grid()
	plt.ylabel('x')
	plt.subplot(311)
	plt.plot(t[:-1],ddx)
	plt.grid()
	plt.ylabel('ddot_x')
	plt.xlabel('Time [s]')

	plt.figure()
	plt.plot(t, inp)
	plt.ylabel('Input [F]')
	plt.xlabel('Time [s]')
	plt.grid()
	plt.show()


#----------- Genetic Programming---------


est_gp = SymbolicRegressor(population_size=8000, 
							generations=4, 
							tournament_size=5, 
							stopping_criteria=1e-4, 
							const_range=(-1, 1),
							init_depth=(2, 5), 
							init_method='half and half', 
							function_set=('add','mul'),# 'sub',),#, 'sqrt', 'sin','div', ) 
							metric='mse', 
							parsimony_coefficient=0.00001, 
							p_crossover=0.80, 
							p_subtree_mutation=0.01, 
							p_hoist_mutation=0.1, 
							p_point_mutation=0.01, 
							p_point_replace=0.05, 
							max_samples=1.0, 
							warm_start=False, 
							n_jobs=1, 
							verbose=1, 
							random_state=None)
#data
#t = np.array(t)
#ddx = np.append(ddx,ddx[-1])


X = np.concatenate((x,dx,inp),axis = 1)
Y = ddx




#fit
est_gp.fit(X, Y)
#print(est_gp._program._depth())


#print(str_est_gp)



#-----------Sympify
locals = {
    'mul': lambda x, y : x*y,
    'add': lambda x, y : x + y
}


str_est_gp = str(est_gp)
eq = sympify(str_est_gp,locals = locals)
print('Clean equation: ', eq)
print('clean equation type:', type(eq))


#plot
my_lib.show_result(est_gp, X, Y, t, plot_show = True)



