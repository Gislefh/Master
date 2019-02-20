import operator
import math
import random

import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from sympy import sympify, cos, sin

import pygraphviz as pgv
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/home/gislehalv/Master/scripts')
import my_lib
from scipy import optimize
from pytictoc import TicToc

## Simulation
time = [0,50,0.05]
mdk = [10, 13, 5]

t,ddx,dx,x,tau = my_lib.MSD(time = time, mdk = mdk, tau = 'square', time_delay = 0.5)

# plt.figure()
# plt.subplot(411)
# plt.plot(t,ddx)
# plt.ylabel('ddx')
# plt.grid()

# plt.subplot(412)
# plt.plot(t,dx)
# plt.ylabel('dx')
# plt.grid()

# plt.subplot(413)
# plt.plot(t,x)
# plt.ylabel('x')
# plt.grid()

# plt.subplot(414)
# plt.plot(t,tau)
# plt.ylabel('tau')
# plt.grid()
# plt.show()
# exit()

##-Add Noise
noise_Y = False
noise_X = False
if noise_Y:
	
	s1 = np.shape(ddx)[0]
	s2 = np.shape(ddx)[1]
	noise_factor = 0.1

	ddx_no_noise = ddx
	ddx = ddx + noise_factor*( np.random.rand(s1,s2) - np.random.rand(s1,s2))

if noise_X:
	s1 = np.shape(ddx)[0]
	s2 = np.shape(ddx)[1]

	n_dx = np.amax(dx)- np.amin(dx)
	n_x = np.amax(x)- np.amin(x)
	n_tau = np.amax(tau)- np.amin(tau)

	noise_factor = 0.3
	
	dx_no_noise = dx
	x_no_noise = x
	tau_no_noise = tau
	dx = dx + 	noise_factor * n_dx * ( np.random.rand(s1,s2) - np.random.rand(s1,s2))
	x = x + 	noise_factor * n_x * ( np.random.rand(s1,s2) - np.random.rand(s1,s2))
	tau = tau + noise_factor * n_tau * ( np.random.rand(s1,s2) - np.random.rand(s1,s2))

	if False:
		plt.figure()

		plt.subplot(311)
		plt.title('Noise Factor = '+str(noise_factor))
		plt.plot(t,dx)
		plt.plot(t,dx_no_noise)
		plt.ylabel('dx [m/s]')
		plt.legend(['dx with added noise', 'original dx'])
		plt.grid()

		plt.subplot(312)
		plt.plot(t,x)
		plt.plot(t,x_no_noise)
		plt.ylabel('x [m]')
		plt.legend(['x with added noise', 'original x'])
		plt.grid()
		
		plt.subplot(313)
		plt.plot(t,tau)
		plt.plot(t,tau_no_noise)
		plt.ylabel('tau [N]')
		plt.xlabel('Time [s]')
		plt.legend(['tau with added noise', 'original tau'])
		plt.grid()	
		
		plt.show()
		#exit()





#Scaling / Preprocessing


#unit variance - divide the variables by their standard deviation 
unit_var = False
if unit_var:
	old_vars = ddx, dx ,x, tau
	std_list = np.std(ddx), np.std(dx), np.std(x), np.std(tau)

	ddx = ddx / np.std(ddx)
	dx = dx / np.std(dx)
	x = x / np.std(x)
	tau = tau / np.std(tau)

	



##Functions
def add3(x,y,z):
	return x+y+z

def mul3(x,y,z):
	return x*y*z

def proDiv(x,y):
	epsilon = 1e-6
	try:
		return x/y 
	except ZeroDivisionError:
		return 1


# Operators
pset = gp.PrimitiveSet("MAIN", 3)
pset.addPrimitive(operator.add, 2)
#pset.addPrimitive(add3, 3)
#pset.addPrimitive(protectedDiv, 2)
#pset.addPrimitive(mul3, 3)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(operator.abs, 1)
pset.addEphemeralConstant("randUn", lambda: random.uniform(-1,1))

#Variable names 
pset.renameArguments(ARG0='dx')
pset.renameArguments(ARG1='x')
pset.renameArguments(ARG2='tau')

# Define a Min Problem
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

##Toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


## Fitness Measurement
MSE = False
RMSE = False
MAE = False
LSR = True

def eval_fit(individual, dx, x, tau, ddx):
	func = toolbox.compile(expr=individual)

	#Mean squared error - seems to give the better results than rmse and mae 
	
	if MSE:
		mse = ((func(dx,x,tau) - ddx )**2)
		return (math.fsum(mse)/len(x),)

	#Root mean squared error
	if RMSE:
		rmse = np.sqrt(math.fsum((func(dx,x,tau) - ddx )**2)/len(tau))
		return (rmse,)

	#Mean absolute error
	if MAE:
		mae = math.fsum(np.abs(func(dx,x,tau) - ddx))/len(tau)
		return (mae,)

	#Least Squares Regression
	if LSR:
		def fun(X):
			tmp = func(X[0]*dx, X[1]*x, X[2]*tau)
			tmp2 = (ddx-tmp)**2
			tmp3 = np.squeeze(tmp2)

			return tmp3

		x0 = np.array([1,1,1])
		sol = optimize.least_squares(fun,x0)

		mse = math.fsum((ddx - func(sol.x[0]*dx, sol.x[1]*x, sol.x[2]*tau))**2)/len(tau)
		return (mse,)




toolbox.register("evaluate", eval_fit,  dx = dx, x = x, tau = tau, ddx = ddx)

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.register("mutate", gp.mutEphemeral, mode = 'all') #mode all: changes the value of all the ephemeral constants
                                                        #mode one: changes the value of one of the individual ephemeral constants
                                                        #seems to work better with this turned on, for the MSD-sys
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))

def main(ddx,dx,x,tau,t):
	random.seed(201)
	pop = toolbox.population(n=400)
	hof = tools.HallOfFame(1)

	stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
	stats_size = tools.Statistics(len)
	mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
	mstats.register("avg", np.mean)
	mstats.register("std", np.std)
	mstats.register("min", np.min)
	mstats.register("max", np.max)

	# implementing noise 
	s1 = np.shape(ddx)[0]
	s2 = np.shape(ddx)[1]

	n_ddx = np.amax(ddx)- np.amin(ddx)
	n_dx = np.amax(dx)- np.amin(dx)
	n_x = np.amax(x)- np.amin(x)
	n_tau = np.amax(tau)- np.amin(tau)

	noise_factor = 0.1

	ddx = ddx 	+ noise_factor * n_ddx*	( np.random.rand(s1,s2) - np.random.rand(s1,s2))
	dx 	= dx  	+ noise_factor * n_dx*	( np.random.rand(s1,s2) - np.random.rand(s1,s2))
	x 	= x 	+ noise_factor * n_x *	( np.random.rand(s1,s2) - np.random.rand(s1,s2))
	tau = tau 	+ noise_factor * n_tau*	( np.random.rand(s1,s2) - np.random.rand(s1,s2))


	#undelay tau
	mse_list = []
	tau_orig = tau.copy()


	scope = np.arange(0,1,0.05)

	for dela in scope:
	
		#re initialize popultion
		pop = toolbox.population(n=400)
		hof = tools.HallOfFame(1)

		stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
		stats_size = tools.Statistics(len)
		mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
		mstats.register("avg", np.mean)
		mstats.register("std", np.std)
		mstats.register("min", np.min)
		mstats.register("max", np.max)

		i = 0
		while dela > t[i]:
			i = i+1
		


		for j in range(len(ddx)):
			if j-i < 0:
				tau[j] = tau_orig[0]
			else:
				tau[j] = tau_orig[j-i]


		## Fit ##
		pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.3, 7, stats=mstats,
									halloffame=hof, verbose=True)

		func = toolbox.compile(expr=hof[0])

		if LSR:
			#find good constants
			def fun(X):
				se = (ddx - func(X[0]*dx, X[1]*x, X[2]*tau))**2
				se_1d = np.squeeze(se)
				return se_1d
			x0 = np.array([1,1,1])
			sol = optimize.least_squares(fun,x0)
					#Add the constants to the equation (new_str)
			tmp = str(hof[0])
			new_str = ''
			skip = 0
			for i in range(len(tmp)):
				if tmp[i] == 'd' and tmp[i+1] == 'x':
					new_str = new_str +"mul(dx,{:.7f})".format(sol.x[0])
					skip = 2

				elif tmp[i] == 'x' and tmp[i-1] != 'd':
					new_str = new_str +"mul(x,{:.7f})".format(sol.x[1])
					skip = 1

				elif tmp[i] == 't' and tmp[i+1] == 'a' and tmp[i+2] == 'u':
					new_str = new_str +"mul(tau,{:.7f})".format(sol.x[2])
					skip = 3

				elif skip == 0:
					new_str = new_str + tmp[i]

				if skip != 0:
					skip = skip - 1

		mse = math.fsum((ddx-func(sol.x[0]*dx, sol.x[1]*x,sol.x[2]*tau))**2)/len(ddx)

		print('delay:',dela,'  mse:', mse)

		mse_list.append(mse)

	plt.figure()
	plt.plot(scope,mse_list)
	plt.grid()
	plt.xlabel('Tested delay [s]')
	plt.ylabel('Resulting MSE')
	plt.show()

	exit()




	return None
	


main(ddx,dx,x,tau,t)
