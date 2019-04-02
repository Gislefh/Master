#sympy simplyfy test
from sympy import sympify, expand
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import operator
import numpy as np


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



test_string = ' - 5.3860039760864065e-6*delta_t*(delta_n + (u**2 + sin(u))*Abs(u)) - 9.705805148682578e-6*delta_t*(delta_n + r + v)'
def expand_sol(sol_string):
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

	### remove constant
	exsp_string = expand(sol_string,locals = locals)
	tot = []
	split = str(exsp_string).split(' ')
	for part in split:

		if part == '+' or part == '-':
			continue
		else:

			split2 = part.split('*')
			tot_part = ''
			for i, part2 in enumerate(split2):
				try:
					float(part2) # is a number
					if part2[i-1] == '': # if the number is a power -> add
						tot_part = tot_part + '*' + part2
				except:
					if tot_part != '':
						tot_part = tot_part + '*' + part2
					else:
						tot_part = part2 #first entry
		
			tot.append(tot_part)

	print(tot)
	# print(exsp_string)
	# test = 'add(delta_n,delta_n)'
	# new_ind = gp.PrimitiveTree.from_string(test,pset)
	# print(new_ind)
	

	### make it into a string for DEAP



	return None


str_ = expand_sol(test_string)







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
