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

	subtree_list = []
	roots = []
	nodes, edges, labels = gp.graph(individual)
	top_root = nodes[0] 
	roots.append([top_root, 0]) #[new root, not been here]

	def split_tree_rec(individual):
		print(roots)
		nodes, edges, labels = gp.graph(individual)

		if labels[0] == 'add' or labels[0] == 'sub': #found new function
			#found root
			if (not [nodes[0],0] in roots) and (not [nodes[0],1] in roots):
				roots.append([nodes[0],0])

			if [nodes[0],0] in roots:
				leg = 1

			elif [nodes[0],1] in roots:
				for level, node in edges[1:]:
					if level == 0:
						leg = node
					else: 
						continue

			else: #should never come here
				return
			print('leg',leg)
			subtree1_root = leg

			F = individual[individual.searchSubtree(subtree1_root)]
			string = ' '
			for item in F:
				if item.arity == 2:
					string = string + item.name + '('				#<-------- needs work for large funcs.
				if item.arity == 0:
					if string[-1] == '(':
						string = string + item.format() +','
					elif string[-1] == ',':
						string = string + item.format() +')'
					else: 
						string = string + item.format()

			new_ind = gp.PrimitiveTree.from_string(string,pset)
			func1 = toolbox.compile(expr=new_ind)
			subtree_list.append(func1)
			print(string)

			if leg == 1:
				for i, item in enumerate(roots):
					if item[0] == nodes[0]:
						roots[i] = [nodes[0],1]	#been here once
			if leg != 1:
				for i, item in enumerate(roots):
					if item[0] == nodes[0]:
						print('remove')
						roots.remove([nodes[0],1])	#been here twice, remowe the listing		


			split_tree_rec(new_ind)
		
		else: #no new function
			print(nodes[0])
			print(individual)
			print(roots)
			exit()

			subtree1_root = nodes[0]
			F = individual[individual.searchSubtree(subtree1_root)]
			string = ' '
			for item in F:
				if item.arity == 2:
					string = string + item.name + '('				#<-------- needs work for large funcs.
				if item.arity == 0:
					if string[-1] == '(':
						string = string + item.format() +','
					elif string[-1] == ',':
						string = string + item.format() +')'
					else: 
						string = string + item.format()

			new_ind = gp.PrimitiveTree.from_string(string,pset)
			func1 = toolbox.compile(expr=new_ind)
			subtree_list.append(func1)

			split_tree_rec(new_ind)


	split_tree_rec(individual)
	print('done')
	print(subtree_list)



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