"""
Genetic Programming on real data. 
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
from scipy.integrate import solve_ivp, cumtrapz
import sys
sys.path.insert(0, '/home/gislehalv/Master/scripts')
import my_lib
from scipy import optimize
from pytictoc import TicToc


# bag path
path = '/home/gislehalv/Master/Data/'
name_train = 'hal_control_2018-12-11-12-13-58_0'
bagFile_path_train = path + name_train + '.bag'

name_val = 'hal_control_2018-12-11-12-19-11_0'
bagFile_path_val = path + name_val + '.bag'


# get data
X = my_lib.open_bag(bagFile_path_train, plot=False)
#exit()
X_val = my_lib.open_bag(bagFile_path_val, plot=False)
"""
X = [u, v, r, du, dv, dr, jet_rpm, nozzle_angle, bucket, interp_arr], interp_arr= time. 
Notes:
- nozzle angle is not the angle but in the range[-100, 100], but the ral angle is in the range[-27, 27] deg
- interp_arr is the time variable 
- bucket shuld be > 95 for all data
"""


solve_for_du = False 
solve_for_dv = True
solve_for_dr = False
if solve_for_du:
	y = X[3]
	y_val = X_val[3]
if solve_for_dv:
	y = X[4]
	y_val = X_val[4]
if solve_for_dr:
	y = X[5]
	y_val = X_val[5]




pset = gp.PrimitiveSet("MAIN", 5)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.abs, 1)
pset.addPrimitive(np.sin, 1)
pset.addPrimitive(np.cos, 1)
#pset.addPrimitive(square, 1)

#Variable names 
pset.renameArguments(ARG0='u')
pset.renameArguments(ARG1='v')
pset.renameArguments(ARG2='r')
pset.renameArguments(ARG3='delta_t')
pset.renameArguments(ARG4='delta_n')


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


#either return_str = True or plot_result = True, not both. 

def eval_fit_new_w_constant(individual, u, v, r, delta_t, delta_n, y, return_str = False, plot_result = False):
	#print('individual: ',individual)
	funcs, str_list = split_tree(individual)
	F_list = []
	

	#top root is not 'add'
	if len(funcs) == 1:

		F = funcs[0](u, v, r, delta_t, delta_n)
		F_trans = np.transpose(F)

		p = np.dot(np.dot(F_trans,F),np.dot(F_trans,y)) 
		p = [p]


	#top root is 'add'
	else:
		for func in funcs:
			F_list.append(func)
		F = np.zeros((len(y), len(F_list)))

		for i, function in enumerate(F_list):
			F[:,i] = np.squeeze(function(u, v, r, delta_t, delta_n))

		F_trans = np.transpose(F)
		try:
			p = np.dot(np.linalg.inv(np.dot(F_trans,F)),np.dot(F_trans,y))  
		except:
			#print('Singular Matrix for: ', individual)
			mse = 1000 # large number
			return(mse,)

	tot_func = np.zeros((len(y)))

	for i, func in enumerate(funcs):
		tot_func = np.add(tot_func, p[i]*func(u, v, r, delta_t, delta_n))


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

	if plot_result:
		plt.figure()
		plt.plot(tot_func)
		plt.plot(y)
		plt.xlabel('Samples')
		plt.legend(['Predicted', 'Ground Truth'])
		plt.grid()


	return(mse,)

toolbox.register("evaluate", eval_fit_new_w_constant, u = X[0], v = X[1], r = X[2], delta_t = X[-4,:], delta_n = X[-3,:], y = y, return_str = False)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))



pop_size = 10000
mate_prob = 0.5
mut_prob = 0.3
generations = 100

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


val_acc = []
train_acc = []

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
	print('Generation:',gen)
	print('Best test set score: ',record['min'])

	train_acc.append(record['min'])

	val_score = eval_fit_new_w_constant(hof[0], u = X_val[0], v = X_val[1], r = X_val[2], delta_t = X_val[-4,:], delta_n = X_val[-3,:], y = y_val, return_str = False)[0]
	val_acc.append(val_score)
	print('validation score: ',val_score)
	

	#save best val- not implemented yet



	#test result on validation set
	if record['min'] < 1e-5:
		mse = eval_fit_new_w_constant(hof[0], u = X_val[0], v = X_val[1], r = X_val[2], delta_t = X_val[-4,:], delta_n = X_val[-3,:], y = y_val, return_str = False)
		print('mse for validation: ', mse)
		if mse[0] < 1e-5:
			#print clean eq, and lisp eq
			print('Final result:',eval_fit_new_w_constant(hof[0], u = X_val[0], v = X_val[1], r = X_val[2], delta_t = X_val[-4,:], delta_n = X_val[-3,:], y = y_val, return_str = True))
			print(hof[0])

			#plot
			eval_fit_new_w_constant(hof[0], u = X_val[0], v = X_val[1], r = X_val[2], delta_t = X_val[-2,:], delta_n = X_val[-4,:], y = y_val, plot_result = True)
			plt.title('Validation set')

			eval_fit_new_w_constant(hof[0], u = X[0], v = X[1], r = X[2], delta_t = X[-2,:], delta_n = X[-4,:], y = y, plot_result = True)
			plt.title('Training set')
			plt.show()
			exit()


print('Reached the max number of generations')
print('Best equation:',eval_fit_new_w_constant(hof[0], u = X_val[0], v = X_val[1], r = X_val[2], delta_t = X_val[-4,:], delta_n = X_val[-3,:], y = y_val, return_str = True))
eval_fit_new_w_constant(hof[0], u = X_val[0], v = X_val[1], r = X_val[2], delta_t = X_val[-4,:], delta_n = X_val[-3,:], y = y_val, plot_result = True)
plt.show()