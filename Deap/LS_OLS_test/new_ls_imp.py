"""
Testing out a new LSR implementation -> as explaned in Genetic Programming and OLS, by amir and amir
Also trying to include OLS
"""




import numpy as np
import operator
import math

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import random
import sys
sys.path.insert(0, '/home/gislehalv/Master/scripts')
import my_lib
from scipy import optimize


## Simulation
time = [0,50,0.1]
mdk = [10, 13, 5]

t,ddx,dx,x,tau = my_lib.MSD(time = time, mdk = mdk, tau = 'square')
t_val,ddx_val,dx_val,x_val,tau_val = my_lib.MSD(time = time, mdk = mdk, tau = 'step')


pset = gp.PrimitiveSet("MAIN", 3)
pset.addPrimitive(operator.add, 2)
#pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
#pset.addPrimitive(operator.neg, 1)
#pset.addPrimitive(operator.abs, 1)
#pset.addPrimitive(np.sin, 1)
#pset.addEphemeralConstant("randUn", lambda: random.randint(-1,1)) ## -no need to add constants if there are no constants in the equation

pset.renameArguments(ARG0='dx')
pset.renameArguments(ARG1='x')
pset.renameArguments(ARG2='tau')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

##Toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


subtree_list = []
def split_tree(individual, ):

	print(individual)
	nodes, edges, labels = gp.graph(individual)
	print('-----------')

	if labels[0] == 'add' or labels[0] == 'sub':
		i = 2
		root = labels[0]
		subtree1_root = 1
		
		
		for level,node in edges[1:]:
			if level == 0:
				subtree2_root = node
			else: 
				continue

		F1 = individual[individual.searchSubtree(subtree1_root)]
		string = ' '
		for item in F1:
			if item.arity == 2:
				string = string + item.name + '('
			if item.arity == 0:
				if string[-1] == '(':
					string = string + item.format() +','
				elif string[-1] == ',':
					string = string + item.format() +')'
				else: 
					string = string + item.format()


		#print(string)
		new_ind = gp.PrimitiveTree.from_string(string,pset)
		func1 = toolbox.compile(expr=new_ind)
		subtree_list.append(func1)
		print(subtree_list)

		split_tree(new_ind)
		exit()

			
			
		

		#sub_tr1 = []
		#sub_tr2 = []		
		#sub_tr1.append(labels[1])
		# for level,node in edges[1:]:
		# 	if level == 0:
		# 		sub_tr2.append(labels[node])
		# 		sub_tr2_root = node
		# 		break
		# 	else:
		# 		sub_tr1.append(labels[node])

		# for level,node in edges[sub_tr2_root:]:
		# 	sub_tr2.append(labels[node])

		# print(sub_tr1)
		# print(sub_tr2)


	"""
	sli = individual.searchSubtree(0)




	#print(type(individual[sli][0]))
	#print(individual[sli][0].format(1,2))
	for i, prim in enumerate(individual):
		#if prim.arity == 0: #terminal node
		#	print(prim.format())
		#if prim.arity == 1: # neg, abs
		#	print(prim.format('lol'))
		if prim.arity == 2 and (prim.name == 'add' or prim.name == 'sub'):
			print(prim.name)



	# string = ""
	# stack = []
	# for node in individual:
	# 	stack.append((node, []))

	# 	while len(stack[-1][1]) == stack[-1][0].arity:
	# 		print(stack)
	# 		prim, args = stack.pop()
	# 		string = prim.format(*args)
	# 		if len(stack) == 0:
	# 			break   # If stack is empty, all nodes should have been seen
	# 		stack[-1][1].append(string)

	
	ind_str = str(individual)

	print(ind_str)
	ind_str = 'mul(1.1,'+ind_str+')'
	print(ind_str)

	a = gp.PrimitiveTree.from_string(ind_str,pset)
	print(a)
	"""
	exit()







old_LSR = False
new_LSR = True
def eval_fit(individual, ddx, dx, x, tau):
	split_tree(individual)
	func = toolbox.compile(expr=individual)
	if old_LSR:
		# old Least Squares Regression
		def fun(X):
			tmp = func(X[0]*dx, X[1]*x, X[2]*tau)
			tmp2 = (ddx-tmp)**2
			tmp3 = np.squeeze(tmp2)

			return tmp3

		x0 = np.array([1,1,1])
		sol = optimize.least_squares(fun,x0)

		mse = math.fsum((ddx - func(sol.x[0]*dx, sol.x[1]*x, sol.x[2]*tau))**2)/len(tau)
		return (mse,)


	if new_LSR:
		# new Least Squares Regression

		a = [1,1,1]
		np.linalg.lstsq()



		#y = ddx
		#F = 

		#p = np.dot(np.dot(np.transpose(F), F), np.dot(np.transpose(F), y)


toolbox.register("evaluate", eval_fit, ddx = ddx, dx = dx, x = x, tau = tau)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))

### main algorithm ##
#constants

pop_size = 500
mate_prob = 0.2
mut_prob = 0.6
generations = 30

#parsimony coefficient
#if MSE_pars:
#	pc = 0.2

population = toolbox.population(n=pop_size)
hof = tools.HallOfFame(1)
logbook = tools.Logbook()

lambda_ = int(pop_size/2)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("max", np.max)
stats.register("min", np.min)


for gen in range(0,generations):
	pop = algorithms.varOr(population, toolbox, lambda_, mate_prob, mut_prob)
	invalid_ind = [ind for ind in population if not ind.fitness.valid]


	fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)	
	for ind, fit in zip(invalid_ind, fitnesses):
		ind.fitness.values = fit
	hof.update(pop)

	record = stats.compile(population)
	logbook.record(gen=gen, evals=len(invalid_ind), **record)
	population = toolbox.select(population, k=len(population))
	print('min: ',record['min'])




















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