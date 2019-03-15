"""
Testing out the new LS alg for the boat sim
Also trying to include OLS
"""

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
from scipy import signal

import pygraphviz as pgv
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/home/gislehalv/Master/scripts')
import my_lib
from scipy import optimize
from pytictoc import TicToc


#time
sim_time = [0, 200, 0.1]


#--inputs--#
def inp_step_x(t,states):
	if t < 5:
		return [[0],[0],[0]]
	else:
		return [[4000],[0],[0]]
def inp_step_series(t,states):
	if t < 5:
		return [[0],[0],[0]]

	elif t < 20:
		return [[4000],[0],[0]]

	elif t < 40:
		return [[0],[1000],[0]]

	elif t < 60:
		return [[0],[0],[1000]]

	elif t < 80:
		return [[0],[0],[-1000]]

	elif t < 100:
		return [[0],[-1000],[1000]]

	elif t < 120:
		return [[2000],[0],[0]]

	elif t < 140:
		return [[0],[0],[-200]]

	elif t < 160:
		return [[300],[0],[0]]

	else:
		return [[0],[0],[0]]
def steps_and_square(t,states):
	if t < 5:
		return [[0],[0],[0]]

	elif t < 20:
		return [[4000],[0],[0]]

	elif t < 40:
		return [[0],[1000],[0]]

	elif t < 60:
		return [[0],[0],[1000]]

	elif t < 80:
		return [[0],[0],[-1000]]

	elif t < 100:
		return [[0],[-1000],[1000]]

	elif t < 120:
		return [[2000],[0],[0]]

	elif t < 140:
		return [[0],[0],[-200]]

	elif t < 160:
		return [[300],[0],[0]]

	elif t < 250:
		return [[0],[1500 * signal.square(t/5)],[700 * signal.square(t/3)]]

	elif t > 270 and t < 300:
		return [[1500 * signal.square(t/3) + 1500],[0],[0]]

	elif t > 320 and t < 400:
		return [[2000 * sin(t/4) +2000],[0],[0]]
	
	elif t > 420 and t < 460:
		return [[2*t],[0],[0]]

	else:
		return [[0],[0],[0]]
def inp_step_x_y_z(t,states):
	if t < 5:
		return [[0],[0],[0]]
	elif t < 55:
		return [[-2000],[0],[0]]
	elif t < 105:
		return [[0],[-500],[0]]
	elif t < 155:
		return [[0],[0],[-200]]	
	elif t < 205:
		return [[1000],[0],[0]]	
	elif t < 255:
		return [[0],[700],[0]]	
	elif t < 305:
		return [[0],[0],[200]]	
	else:
		return [[0],[0],[0]]


#WJ input
prev_jet_input = []
prev_noz_input = []
prev_t = []

def input_WJ(t, states):
	# def jet_model(nu, jet_rpm, delta_nozzle):
	# 	#constants
	# 	lever_CGtowj_port = [-3.82, -0.475]
	# 	lever_CGtowj_stb = [-3.82, 0.475]
	# 	rpm_slew_rate = [2000, -2000]
	# 	nozzle_slew_rate = [1.3464, -1.3464]
	# 	rpm_min_max = [0, 2000]

	# 	if jet_rpm > rpm_min_max[1]:
	# 		jet_rpm = rpm_min_max[1]
	# 	elif jet_rpm < rpm_min_max[0]:
	# 		jet_rpm = rpm_min_max[0]

		
	# 	if 0:#prev_jet_input and prev_noz_input:
	# 		#rate limiter rpm
	# 		prev_jet_input.append(jet_rpm)
	# 		jet_now = prev_input[-1]
	# 		jet_prev = prev_input[-2]

	# 		rate = (jet_now - jet_prev)/(t_now-t_prev)
	# 		if rate > rpm_slew_rate[0]:
	# 			new_rpm = (t_now-t_prev)*rpm_slew_rate[0] + jet_prev
	# 		elif rate < rpm_slew_rate[1]:
	# 			new_rpm = (t_now-t_prev)*rpm_slew_rate[1] + jet_prev

	# 		#rate limiter nozzle
	# 		prev_noz_input.append(delta_nozzle)
	# 		jet_now = prev_noz_input[-1]
	# 		jet_prev = prev_noz_input[-2]
	# 		rate = (noz_now - noz_prev)/(t_now-t_prev)

	# 		if rate > nozzle_slew_rate[0]:
	# 			new_rpm = (t_now-t_prev)*nozzle_slew_rate[0] + jet_prev
	# 		elif rate < nozzle_slew_rate[1]:
	# 			new_rpm = (t_now-t_prev)*nozzle_slew_rate[1] + jet_prev





	# 	#rpm2thrust
	# 	speed = nu[0] * 1.94384 # knots 
	# 	a0 = 6244.15
	# 	a1 = -178.46
	# 	a2 = 0.881043
	# 	thrust_unscaled = a0 + a1*speed + a2*(speed**2)

	# 	r0 = 85.8316
	# 	r1 = -1.7935
	# 	r2 = 0.00533
	# 	rpm_scale = 1/4530*(r0 + r1*jet_rpm + r2 * (jet_rpm **2))

	# 	thrust = rpm_scale * thrust_unscaled


	# 	#waterjet port
	# 	#force
	# 	Fx = thrust*np.cos(delta_nozzle)
	# 	Fy = thrust*np.sin(delta_nozzle)
	# 	#moment
	# 	Nz_port = (lever_CGtowj_port[0]*Fy)- (lever_CGtowj_port[1]*Fx)
	# 	Nz_stb = (lever_CGtowj_stb[0]*Fy)- (lever_CGtowj_stb[1]*Fx)

	# 	#tau_b_port = [Fx, Fy, Nz_port]
	# 	#tau_b_stb = [Fx, Fy, Nz_stb]

	# 	tau_b =  [2*Fx, 2*Fy, Nz_port + Nz_stb]#np.add(tau_b_port, tau_b_stb)
	# 	prev_jet_input.append(jet_rpm)
	# 	prev_noz_input.append(delta_nozzle)
	# 	return tau_b


	nu = states[3:6]
	jet_rpm = 500 
	if t < 18:
		delta_nozzle = 0
	elif t < 20:
		delta_nozzle = 0.2
		jet_rpm = 0
	else:
		delta_nozzle = 0
		jet_rpm = 100
	prev_t.append(t)
	tau_b = my_lib.jet_model(nu, jet_rpm, delta_nozzle)


	return tau_b



### RUN SIM  ###

X = my_lib.boat_simulation(input_WJ, time = sim_time)
#X_val = my_lib.boat_simulation(inp_step_series, time = sim_time)

my_lib.boat_sim_plot(X, show = False)

inputs = np.zeros((len(prev_jet_input), 2))
inputs[:,0] = prev_jet_input
inputs[:,1] = prev_noz_input
plt.figure()
plt.plot(prev_t, inputs[:, 0])
plt.plot(prev_t, inputs[:, 1])
plt.grid()
plt.legend(['jet_rpm', 'jet_nozzle'])
plt.show()
exit()

###  what eq to find.
solve_for_du = False 
solve_for_dv = False
solve_for_dr = True
if solve_for_du:
	y = X[3]
	y_val = X_val[3]
if solve_for_dv:
	y = X[4]
	y_val = X_val[4]
if solve_for_dr:
	y = X[5]
	y_val = X_val[5]


#Operators
pset = gp.PrimitiveSet("MAIN", 6)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.abs, 1)

#Variable names 
pset.renameArguments(ARG0='u')
pset.renameArguments(ARG1='v')
pset.renameArguments(ARG2='r')
pset.renameArguments(ARG3='tau_x')
pset.renameArguments(ARG4='tau_y')
pset.renameArguments(ARG5='tau_z')


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

##Toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


#works for arity 0, 1 and 2 and only for add (not sub)
def split_tree(individual):
	
	def tree_trav(individual):
		nodes, edges, labels = gp.graph(individual)
		main_roots = []

		#is the first root add or sub
		if labels[0] == 'add':# or labels[0] == 'sub':
			main_roots.append(nodes[0])
		else:
			return None

		#find the main roots
		for node in sorted(nodes):
			if labels[node] == 'add':# or labels[node] == 'sub':
				if node not in main_roots:
					for edge in edges: 
						if node == edge[1] and edge[0] in main_roots: #if the previus node is in roots
							main_roots.append(node)

		for root in main_roots:
			for edge in edges:
				if edge[0] in main_roots:
					if edge[1] not in main_roots and edge[1] not in roots:					
						roots.append(edge[1])
		return main_roots

	def ext_funcs(individual):
		for root in roots:

			#divide into substrings
			F = individual[individual.searchSubtree(root)]

			string = ''


			if len(F) == 1:
				string = F[0].format()

			else:		
	
				for item in F:
					if item.arity == 0:
						string = string.split(' ')[0] + item.format() + ' '.join(string.split(' ')[1:])
					if item.arity == 1:
						if len(string) > 0:
							string = string.split(' ')[0] + item.name + '( )' + ' '.join(string.split(' ')[1:])
						else:
							string = item.name + '( )'

					if item.arity == 2:
						if len(string) > 0:
							string = string.split(' ')[0] + item.name + '( , )' + ' '.join(string.split(' ')[1:])
						else:
							string = item.name + '( , )' 


			str_list.append(string)
			new_ind = gp.PrimitiveTree.from_string(string,pset)
			func1 = toolbox.compile(expr=new_ind)
			subtree_list.append(func1)


	subtree_list = []
	str_list = []
	roots = []
	main_roots = tree_trav(individual)
	if main_roots == None:
		str_list.append(str(individual))
		return [toolbox.compile(expr=individual)], str_list

	ext_funcs(individual)
	return subtree_list, str_list

def eval_fit_new(individual, u, v, r, tau_x, tau_y, tau_z, y, return_str = False):
	#print('individual: ',individual)
	funcs, str_list = split_tree(individual)
	F_list = []
	

	#top root is not 'add'
	if len(funcs) == 1:

		F = funcs[0](u,v,r,tau_x,tau_y,tau_z)
		F_trans = np.transpose(F)

		p = np.dot(np.dot(F_trans,F),np.dot(F_trans,y)) 
		p = [p]


	#top root is 'add'
	else:
		for func in funcs:
			F_list.append(func)
		F = np.zeros((len(y), len(F_list)))

		for i, function in enumerate(F_list):
			F[:,i] = np.squeeze(function(u,v,r,tau_x,tau_y,tau_z))

		F_trans = np.transpose(F)
		try:
			p = np.dot(np.linalg.inv(np.dot(F_trans,F)),np.dot(F_trans,y))  
		except:
			#print('Singular Matrix for: ', individual)
			mse = 1000 # large number
			return(mse,)

	tot_func = np.zeros((len(y)))

	for i, func in enumerate(funcs):
		tot_func = np.add(tot_func, p[i]*func(u,v,r,tau_x,tau_y,tau_z))


	mse = math.fsum((y-tot_func)**2)/len(y)


	#return the simplified eq
	if return_str:
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
		tot_str = ''
		for i, func_str in enumerate(str_list):
			tot_str = tot_str +'+'+ str(p[i])+ '*' +func_str
		function_string = sympify(tot_str,locals = locals)
		return function_string

	return(mse,)



#test function
if 0:
	individual = 'add(add(u,v),mul(r,tau_z))'
	individual = gp.PrimitiveTree.from_string(individual,pset)
	eval_fit_new(individual, u = X[0], v = X[1], r = X[2], tau_x = X[6], tau_y = X[7], tau_z = X[8], du = X[3], return_str = False)
	print(eval_fit_new(individual, u = X[0], v = X[1], r = X[2], tau_x = X[6], tau_y = X[7], tau_z = X[8], du = X[3], return_str = True))
	exit()




toolbox.register("evaluate", eval_fit_new, u = X[0], v = X[1], r = X[2], tau_x = X[6], tau_y = X[7], tau_z = X[8], y = y, return_str = False)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))

### main algorithm ##
#constants

pop_size = 5000
mate_prob = 0.5
mut_prob = 0.3
generations = 30

#parsimony coefficient
#if MSE_pars:
#	pc = 0.2

pop = toolbox.population(n=pop_size)
hof = tools.HallOfFame(1)
logbook = tools.Logbook()

lambda_ = int(pop_size/2)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("max", np.max)
stats.register("min", np.min)


for gen in range(0,generations):
	pop = algorithms.varOr(pop, toolbox, lambda_, mate_prob, mut_prob)
	invalid_ind = [ind for ind in pop if not ind.fitness.valid]

	#print(len(invalid_ind))
	fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)	
	for ind, fit in zip(invalid_ind, fitnesses):
		ind.fitness.values = fit
	hof.update(pop)

	record = stats.compile(pop)
	logbook.record(gen=gen, evals=len(invalid_ind), **record)
	pop = toolbox.select(pop, k=len(pop))
	print('Best test set score: ',record['min'])

	print('validation score: ',eval_fit_new(hof[0], u = X_val[0], v = X_val[1], r = X_val[2], tau_x = X_val[6], tau_y = X_val[7], tau_z = X_val[8], y = y_val, return_str = False)[0])
	

	#test result on validation set
	if record['min'] < 1e-8:
		mse = eval_fit_new(hof[0], u = X_val[0], v = X_val[1], r = X_val[2], tau_x = X_val[6], tau_y = X_val[7], tau_z = X_val[8], y = y_val, return_str = False)
		print('mse for validation: ', mse)
		if mse[0] < 1e-8:
			print('eq: ', hof[0])
			print('Final result:',eval_fit_new(hof[0], u = X_val[0], v = X_val[1], r = X_val[2], tau_x = X_val[6], tau_y = X_val[7], tau_z = X_val[8], y = y_val, return_str = True))
			print(hof[0])
			break


##history plot
# plt.figure()
# plt.semilogy(logbook.select('gen'),logbook.select('min'))
# plt.xlabel('Generations')
# plt.ylabel('Mean Sqaure Error')
# plt.grid()
# plt.show()

















"""

	#terminate? - check with validation data
	if record['min'] < 1e-6:
		func = toolbox.compile(expr=hof[0])
		#LSR - to find weights
		def fun(X):
			tmp = func(X[0]*dx, X[1]*x, X[2]*tau)
			tmp2 = (ddx-tmp)**2
			tmp3 = np.squeeze(tmp2)

			return tmp3

		x0 = np.array([1,1,1])
		sol = optimize.least_squares(fun,x0)

		#MSE validation data
		func = toolbox.compile(expr=hof[0])
		mse_val = math.fsum((ddx_val - func(sol.x[0]*dx_val, sol.x[1]*x_val, sol.x[2]*tau_val))**2)/len(tau)

		if mse_val < 1e-5:
			break
		else:
			new_str, sol = my_lib.new_string_from_LSR(func, hof[0], ddx, dx, x, tau)
			print('solution that did not validate well: ', new_str)





func = toolbox.compile(expr=hof[0])	
new_str, sol = my_lib.new_string_from_LSR(func, hof[0], ddx, dx, x, tau)
	
print(new_str)
"""