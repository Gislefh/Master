"""
test with chosen string to divide the individual into subfunctions to weigth

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

if 1:
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




def split_tree(individual):
	
	def tree_trav(individual):
		nodes, edges, labels = gp.graph(individual)
		main_roots = []

		#is the first root add or sub
		if labels[0] == 'add' or labels[0] == 'sub':
			main_roots.append(nodes[0])
		else:
			return None

		#find the main roots
		for node in sorted(nodes):
			if labels[node] == 'add' or labels[node] == 'sub':
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

			F = individual[individual.searchSubtree(root)]
			string = ' '
			for item in F:
				if item.arity == 2:
					if string[-1] == ')':
						string = string + ',' + item.name + '('	
					elif string[-1] == ' ':
						string = string + item.name + '('
					else:
						string = string + item.name +'('

				if item.arity == 0:
					if string[-1] == '(':
						string = string + item.format() +','
					elif string[-1] == ',':
						string = string + item.format() +')'
					elif string[-1] == ')':
						string = string +','+ item.format() +')'
					else: 
						string = string + item.format()

			print(string)
			new_ind = gp.PrimitiveTree.from_string(string,pset)
			func1 = toolbox.compile(expr=new_ind)
			subtree_list.append(func1)
			


	subtree_list = []
	roots = []
	main_roots = tree_trav(individual)
	ext_funcs(individual)

	for func in subtree_list:
		print(func(1,0,1))



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

if 1:
	toolbox.register("evaluate", eval_fit, ddx = ddx, dx = dx, x = x, tau = tau)
	toolbox.register("select", tools.selTournament, tournsize=5)
	toolbox.register("mate", gp.cxOnePoint)
	toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
	toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
	toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
	toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
				

test_str = gp.PrimitiveTree.from_string('add(add(dx, x), add(tau, x))',pset)


split_tree(test_str, )