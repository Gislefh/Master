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

#not working for windows :(
plot_tree = False
if plot_tree:
	import pygraphviz as pgv
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/home/gislehalv/Master/scripts')
import my_lib
from scipy import optimize
from pytictoc import TicToc

## Simulation
time = [0,50,0.1]
mdk = [10, 13, 5]

t,ddx,dx,x,tau = my_lib.MSD(time = time, mdk = mdk, tau = 'square')



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

def main(gen_len = 1, pop_size = 1, verb = True):
	#random.seed(200)
	pop = toolbox.population(n=pop_size)
	hof = tools.HallOfFame(1)

	stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
	stats_size = tools.Statistics(len)
	mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
	mstats.register("avg", np.mean)
	mstats.register("std", np.std)
	mstats.register("min", np.min)
	mstats.register("max", np.max)

	## Fit ##
	pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.3, gen_len, stats=mstats,
								halloffame=hof, verbose=verb)


	func = toolbox.compile(expr=hof[0])

	# ### LSR ####
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


	# ### Graphviz Tree ###
	if plot_tree:
		expr = toolbox.individual()
		nodes, edges, labels = gp.graph(hof[0])

		#Add input weights to the tree
		if LSR:
			for i in nodes:
				if isinstance(labels[i], str):
					if labels[i] == 'dx':
						labels[i] = 'dx*'+"{:.3f}".format(sol.x[0])
					if labels[i] == 'x':
						labels[i] = 'x*'+"{:.3f}".format(sol.x[1])
					if labels[i] == 'tau':
						labels[i] = 'tau*'+"{:.3f}".format(sol.x[2])
				elif isinstance(labels[i], float):
					labels[i] = "{:.3f}".format(labels[i])

		g = pgv.AGraph()
		g.add_nodes_from(nodes)
		g.add_edges_from(edges)
		g.layout(prog="dot")

		for i in nodes:
			n = g.get_node(i)
			n.attr["label"] = labels[i]

		g.draw("tree.pdf")



	####### Sympify #######
	locals = {
		'mul': lambda x, y : x * y,
		'add': lambda x, y : x + y,
		'add3': lambda x, y, z: x+y+z,
		'sub': lambda x, y : x - y,
		'protectedDiv': lambda x, y: x / y,
		'neg': lambda x: -x,
		'sin': lambda x: sin(x),
		'cos': lambda x: cos(x),
		'abs': lambda x: np.abs(x)#x if x >= 0 else -x
	}



	hof_str = str(hof[0])
	if LSR:
		eq = sympify(new_str,locals = locals)
	else:
		eq = sympify(hof_str,locals = locals)
	


	# if unit_var: #scaling
	# 	mS_ddx = mdk[0]*std_list[0] 
	# 	#print('Scaled eq: ddx_scaled =',std_list[-1]/mS_ddx,'*tau -',(mdk[2]*std_list[2])/mS_ddx,'*x - ',(mdk[1]*std_list[1])/mS_ddx,'*dx'  )
	# else:
	# 	#print('True eq: ddx =', 1/mdk[0],'*tau -',mdk[2]/mdk[0],'*x -',mdk[1]/mdk[0],'*dx')   	
	# #print('Clean equation: ', eq)
	# #print('LISP with no input weights:', hof_str)


	#### Accuracy with noise ###
	if noise_Y and not noise_X:
		if LSR:
			mse_result = math.fsum((ddx_no_noise - func(sol.x[0]*dx, sol.x[1]*x,sol.x[2]*tau))**2)/len(tau)
		else:
			mse_result = math.fsum((ddx_no_noise-func(dx,x,tau))**2)/len(tau)
		#print('The final MSE of the solution, without noise: ',mse_result)

	elif noise_X and not noise_Y: 
		if LSR:
			mse_result = math.fsum((ddx - func(sol.x[0]*dx_no_noise, sol.x[1]*x_no_noise,sol.x[2]*tau_no_noise))**2)/len(tau)
		else:
			mse_result = math.fsum((ddx-func(dx_no_noise,x_no_noise,tau_no_noise))**2)/len(tau)
		#print('The final MSE of the solution, without noise: ',mse_result)

	elif noise_X and noise_Y: 
		if LSR:
			mse_result = math.fsum((ddx_no_noise - func(sol.x[0]*dx_no_noise, sol.x[1]*x_no_noise,sol.x[2]*tau_no_noise))**2)/len(tau)
		else:
			mse_result = math.fsum((ddx_no_noise-func(dx_no_noise,x_no_noise,tau_no_noise))**2)/len(tau)
		#print('The final MSE of the solution, without noise: ',mse_result)

	else:
		if LSR:
			mse_result = math.fsum((ddx - func(sol.x[0]*dx, sol.x[1]*x,sol.x[2]*tau))**2)/len(tau)
		else:
			mse_result = math.fsum((ddx-func(dx,x,tau))**2)/len(tau)
		#print('The final MSE of the solution, without noise: ',mse_result)


	# #####  Plot Solution#####
	if 0:
		plt.figure(1)
		plt.title('MSE = '+'{:.4f}'.format(mse_result*1e6) + '1e-6')
		if LSR:
			plt.plot(t,func(sol.x[0]*dx, sol.x[1]*x,sol.x[2]*tau))
		else:
			plt.plot(t,func(dx,x,tau))
		plt.plot(t,ddx, linewidth=0.6)
		plt.xlabel('Time [s]')
		plt.ylabel('ddx [m/s²]')
		plt.grid()

		if noise_Y and not noise_X:
			plt.plot(t,ddx_no_noise)
			plt.legend(['prediction','ddx seen by the algorithm', 'ddx without noise'])

		elif noise_X and not noise_Y:
			plt.plot(t,func(sol.x[0]*dx_no_noise, sol.x[1]*x_no_noise, sol.x[2]*tau_no_noise))
			plt.legend(['prediction','ddx ', 'prediction without noise'])

		elif noise_X and noise_Y:
			plt.plot(t,ddx_no_noise)
			plt.plot(t,func(sol.x[0]*dx_no_noise, sol.x[1]*x_no_noise, sol.x[2]*tau_no_noise))
			plt.legend(['prediction','ddx seen by the algorithm', 'ddx without noise', 'prediction without noise'])

		else:
			plt.legend(['prediction','actual equation'])

	
	return log, pop_size, gen_len, mse_result



#STATS - generations vs population size
N_gen = [2, 5, 10, 20, 50, 100, 200]
#N_pop = [5, 20, 50, 200, 500, 2000]

t = TicToc()
mse_list = []
cor_sol_list = []
for enum, j in enumerate(N_gen):

	avr_time = 0
	avr_mse_result = 0
	n_correct_sol = 0

	for i in range(10):
		print('Ngen: ', j,' itter: ',i, end = ' ')
		t.tic()
		log_book, pop_size, gen_len, mse_result = main(gen_len = j, pop_size = 500, verb = False)
		avr_time = avr_time + t.tocvalue()
		avr_mse_result = avr_mse_result + mse_result

		gen = log_book.select("gen")
		fit_mins = log_book.chapters["fitness"].select("min")

		plt.figure(2+enum)
		plt.semilogy(gen, fit_mins, 'ro-')


		if mse_result < 1e-7:
			print('Sol found')
			n_correct_sol = n_correct_sol + 1
		else:
			print('Sol not found')

	cor_sol_list.append(n_correct_sol)
	mse_list.append(avr_mse_result/10)
	plt.title('History - Population Size: '+str(pop_size)+', Number of Generations: ' + str(j)+ ', Average Run Time: '+'{0:.2f}s'.format(avr_time/10))
	plt.xlabel('Generations')
	plt.ylabel('Min Fitness')
	plt.grid()
print('correct solution list: ',cor_sol_list )
print('mse_list:',mse_list)
plt.show()
