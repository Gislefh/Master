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

from sympy import sympify, cos, sin, expand, collect, Lambda, lambdify, symbols
from scipy import signal, linalg

import pygraphviz as pgv
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumtrapz
import sys
sys.path.insert(0, '/home/gislehalv/Master/scripts')
import my_lib
from scipy import optimize
from pytictoc import TicToc

bag_1 = 'hal_control_2018-12-11-10-53-26_0' #large!
bag_2 = 'hal_control_2018-12-11-11-49-22_0' #similar to bag1 but smaller
bag_3 = 'hal_control_2018-12-11-12-13-58_0' #
bag_4 = 'hal_control_2018-12-11-12-19-11_0'





# ####test
# new_func_str = ['a', 'b', 'c', 'd', 'e']
# print(new_func_str)
# for h in range(int(np.ceil(np.log2(len(new_func_str))))): #
# 	tmp = []
# 	i = 0
# 	while i < len(new_func_str):
# 		if i +1 < len(new_func_str):
# 			tmp.append('add(' + new_func_str[i] + ',' + new_func_str[i+1] + ')')
# 			i = i+2
# 		else:
# 			tmp.append(new_func_str[i])
# 			i = i+1
# 		#print(tmp)
# 	new_func_str = tmp
# 	print(new_func_str)
			
# exit()


##

# bag path
path = '/home/gislehalv/Master/Data/'


bagFile_path_train = path + bag_3 + '.bag'

bagFile_path_val = path + bag_2 + '.bag'
bagFile_path_test = path + bag_3 + '.bag'


# get data
X = my_lib.open_bag(bagFile_path_train, plot=False, thr_bucket = False, filter_cutoff = 0.1)


X_val = my_lib.open_bag(bagFile_path_val, plot=False, thr_bucket = False, filter_cutoff = 0.1)
X_test = my_lib.open_bag(bagFile_path_test, plot=False, thr_bucket = False, filter_cutoff = 0.1)


"""
X = [u, v, r, du, dv, dr, jet_rpm, nozzle_angle, bucket, interp_arr], interp_arr= time. 
Notes:
- nozzle angle is not the angle but in the range[-100, 100], but the ral angle is in the range[-27, 27] deg
- interp_arr is the time variable 
- bucket shuld be > 95 for all data
"""


###--scaling
#scaling to zero mean

scaling = False
if scaling:
	X_orig = X.copy()
	X_val_orig = X_val.copy()
	for i in range(9):
		X[i] = X[i]  / np.std(X[i])
		X_val[i] = X_val[i] /np.std(X_val[i])


#test for du
test_du = False
if test_du:
	x3_std = np.std(X[3])
	x3_val_std = np.std(X_val[3])
	print(x3_std)
	X[3] = X[3] / np.std(X[3])
	X_val[3] = X_val[3] / np.std(X_val[3])


## what variable to use as y
solve_for_du = True 
solve_for_dv = False
solve_for_dr = False
if solve_for_du:
	y = X[3]
	y_val = X_val[3]
	y_test = X_test[3]
if solve_for_dv:
	y = X[4]
	y_val = X_val[4]
	y_test = X_test[4]
if solve_for_dr:
	y = X[5]
	y_val = X_val[5]
	y_test = X_test[5]




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

def OLS(individual, u, v, r, delta_t, delta_n, y, threshold = 0.05):
	funcs, str_list = split_tree(individual)
	F_list = []

	#top root is not 'add'
	if len(funcs) == 1:

		#F = funcs[0](u, v, r, delta_t, delta_n)
		return individual

	#top root is 'add'
	else:
		for func in funcs:
			F_list.append(func)
		F = np.zeros((len(y), len(F_list)))

		for i, function in enumerate(F_list):
			F[:,i] = np.squeeze(function(u, v, r, delta_t, delta_n))

	#e = y-np.dot(F,p)
	q,r_ = np.linalg.qr(F, mode = 'reduced')
	d = np.dot(np.transpose(q),q)
	g = np.dot(np.linalg.inv(d),np.dot(np.transpose(q),y))
	err = np.zeros((len(g)))
	for i in range(len(g)):
		err[i] = len(y) * ((g[i]**2 * np.dot(np.transpose(q[i,:]),q[i,:]) )/np.dot(np.transpose(y),y))


	#sort
	str_list_sort = [x for _,x in sorted(zip(err,str_list))]
	funcs = [x for _,x in sorted(zip(err,funcs))]
	err_sort = np.sort(err)
	#print('err: ', err_sort)
	#print('funcs: ', str_list_sort)

	#remove funcs that does not contribute to the fitness
	tot_func = np.zeros((len(y)))
	new_funcs = []
	new_func_str = []
	for i, error_red in enumerate(err_sort):
		if error_red < threshold:
			x=1
			#print('removed:', str_list_sort[i])
		else:
			new_funcs.append(funcs[i])
			new_func_str.append(str_list_sort[i])

	#if no functions remaining, return the individual -> not ideal. FIX
	if not new_funcs:
		return individual

	show_new_mse = False
	if show_new_mse:
		#find new params.
		F_list = []
		if len(new_funcs) == 1:

			F = new_funcs[0](u, v, r, delta_t, delta_n)
			F_trans = np.transpose(F)

			p = np.dot(np.dot(F_trans,F),np.dot(F_trans,y)) 
			p = [p]


		else:
			for func in new_funcs:
				F_list.append(func)
			F = np.zeros((len(y), len(F_list)))

			for i, function in enumerate(F_list):
				F[:,i] = np.squeeze(function(u, v, r, delta_t, delta_n))


			F_trans = np.transpose(F)  
			try:
				p = np.dot(np.linalg.inv(np.dot(F_trans,F)),np.dot(F_trans,y))  
			except:
				print('Singular Matrix for: ', individual)
				mse = 1000 # large number
				return(mse,)

		#mse
		for i, func in enumerate(new_funcs):
			tot_func = np.add(tot_func, p[i]*func(u, v, r, delta_t, delta_n))
		print('new mse: ',math.fsum((y-tot_func)**2)/len(y))

	
	for h in range(int(np.ceil(np.log2(len(new_func_str))))): #
		tmp = []
		i = 0
		while i < len(new_func_str):
			if i +1 < len(new_func_str):
				tmp.append('add(' + new_func_str[i] + ',' + new_func_str[i+1] + ')')
				i = i+2
			else:
				tmp.append(new_func_str[i])
				i = i+1
		new_func_str = tmp
		#print(new_func_str)	

	new_ind = gp.PrimitiveTree.from_string(new_func_str[0],pset)
	new_func = creator.Individual(new_ind)
	return new_func



def eval_fit_new_w_constant(individual, u, v, r, delta_t, delta_n, y, return_str = False, plot_result = False, return_function = False):
	#print('individual: ',individual)
	funcs, str_list = split_tree(individual)
	F_list = []
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
	expand_ind = False
	if expand_ind:
		def expand_functions(exp_str):
			u, v, r, delta_t, delta_n = symbols('u v r delta_t delta_n')
			for sub_func in range(len(exp_str.args)):
				f = lambdify((u,v,r, delta_t, delta_n),exp_str.args[sub_func], 'numpy')
				new_funcs.append(f)
				new_str_list.append(str(exp_str.args[sub_func]))


		##### check if it can be expanded further with LIP
		new_funcs = []
		new_str_list = []

		for i, part in enumerate(str_list):
			#sympify and expand
			simp_str = sympify(part,locals = locals)
			exp_str = expand(simp_str,locals)
			
			if str(simp_str) != str(exp_str): #can be expanded further -> create new functions
				expand_functions(exp_str)
				
			else: #can't -> add old functions
				new_funcs.append(funcs[i])
				new_str_list.append(str(exp_str))



		funcs = new_funcs
		str_list = new_str_list

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
		if expand_ind:
			tot_str = ''
			for i, func_str in enumerate(str_list):
				if p[i] < 0:
					tot_str = tot_str +' '+ str(p[i])+ '*' +func_str
				else:
					tot_str = tot_str + ' + ' + str(p[i])+ '*' +func_str
			function_string = sympify(tot_str,locals = locals)
			return tot_str

		elif 0:
			tot_str = ''
			for i, func_str in enumerate(str_list):
				tot_str = tot_str +'+'+ str(p[i])+ '*' +str(sympify(func_str,locals = locals))
			function_string = tot_str
			return function_string
		else:
			tot_str = ''
			for i, func_str in enumerate(str_list):
				tot_str = tot_str +'+'+ str(p[i])+ '*' +func_str
			function_string = str(sympify(tot_str,locals = locals))
			return function_string

	if plot_result:
		#assuming 0.05s steps
		time = np.arange(0,np.ceil(len(y)/0.05), 0.05)
		plt.figure()
		plt.plot(time[:len(y)], tot_func)
		plt.plot(time[:len(y)], y)
		#plt.plot(time[:len(y)], tot)
		plt.xlabel('Time [s]')
		plt.legend(['Predicted', 'Ground Truth'])
		plt.grid()

	if return_function:
		tot_str = ''
		for i, func_str in enumerate(str_list):
			tot_str = tot_str +'+'+ str(p[i])+ '*' +func_str
		function_string = str(sympify(tot_str,locals = locals))
		def create_function(str_list_):
			u, v, r, delta_t, delta_n = symbols('u v r delta_t delta_n')
			tmp_f = str(sympify(str_list_,locals = locals))
			list_ = lambdify((u,v,r, delta_t, delta_n), tmp_f, 'numpy')
			return list_
		lam_func = create_function(function_string)
		return lam_func


	return(mse,)

toolbox.register("evaluate", eval_fit_new_w_constant, u = X[0], v = X[1], r = X[2], delta_t = X[-4,:], delta_n = X[-3,:], y = y, return_str = False)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=3, max_=4)  #<-------- gives faster results with 2,4 than 0,2 'perhaps'
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))



pop_size = 10000
mate_prob = 0.5
mut_prob = 0.3
generations = 30
ols_thr = 0.1


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
best_val = 10000

for gen in range(0,generations):
	pop = algorithms.varOr(pop, toolbox, lambda_, mate_prob, mut_prob)
	invalid_ind = [ind for ind in pop if not ind.fitness.valid]

	# for i, individual in enumerate(invalid_ind):
	# 	invalid_ind[i] = OLS(individual, u = X[0], v = X[1], r = X[2], delta_t = X[-4,:], delta_n = X[-3,:], y = y, threshold = ols_thr)


	#print(len(invalid_ind))
	fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)	
	for ind, fit in zip(invalid_ind, fitnesses):
		ind.fitness.values = fit
	hof.update(pop)


	record = stats.compile(pop)
	logbook.record(gen=gen, evals=len(invalid_ind), **record)
	pop = toolbox.select(pop, k=len(pop))

		
	
	
	train_score = eval_fit_new_w_constant(hof[0], u = X[0], v = X[1], r = X[2], delta_t = X[-4,:], delta_n = X[-3,:], y = y, return_str = False)[0]
	func = eval_fit_new_w_constant(hof[0], u = X[0], v = X[1], r = X[2], delta_t = X[-4,:], delta_n = X[-3,:], y = y, return_str = False, return_function = True)
	val_score = math.fsum((y_val-func(X_val[0], X_val[1], X_val[2], X_val[-4], X_val[-3]))**2)/len(y_val)
	
	print('Generation:',gen)
	print('training score: ',train_score)
	print('validation score: ',val_score)
	
	val_acc.append(val_score)
	train_acc.append(train_score)

	#save best val 
	if  val_score < best_val:
		best_val_ind = hof[0]
		best_val = val_score
		print('Saved as new best')
	continue
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
print('Best equation:',eval_fit_new_w_constant(best_val_ind, u = X[0], v = X[1], r = X[2], delta_t = X[-4,:], delta_n = X[-3,:], y = y, return_str = True))


eval_fit_new_w_constant(best_val_ind, u = X[0], v = X[1], r = X[2], delta_t = X[-4,:], delta_n = X[-3,:], y = y, plot_result = True)
plt.title('Training set')

func = eval_fit_new_w_constant(best_val_ind, u = X[0], v = X[1], r = X[2], delta_t = X[-4,:], delta_n = X[-3,:], y = y, return_str = False, return_function = True)

#plot val and train set

plt.figure()
plt.plot(X_val[-1], func(X_val[0], X_val[1], X_val[2], X_val[-4], X_val[-3]))
plt.plot(X_val[-1], y_val)
plt.grid()
plt.title('Validation set')
plt.xlabel('Time [s]')
plt.ylabel('du')

plt.figure()
plt.plot(X_test[-1], func(X_test[0], X_test[1], X_test[2], X_test[-4], X_test[-3]))
plt.plot(X_test[-1], y_test)
plt.grid()
plt.title('Test set')
plt.xlabel('Time [s]')
plt.ylabel('du')



#rescale back 
if scaling:
	X = X_orig.copy()
	X_val = X_val_orig.copy()

	eval_fit_new_w_constant(best_val_ind, u = X_val[0], v = X_val[1], r = X_val[2], delta_t = X_val[-4,:], delta_n = X_val[-3,:], y = y_val, plot_result = True)
	plt.title('Validation set, rescaled')

	eval_fit_new_w_constant(best_val_ind, u = X[0], v = X[1], r = X[2], delta_t = X[-2,:], delta_n = X[-4,:], y = y, plot_result = True)
	plt.title('Training set, rescaled')

#test for du
if test_du:
	y = y * x3_std
	y_val = y_val * x3_val_std


	eval_fit_new_w_constant(best_val_ind, u = X_val[0], v = X_val[1], r = X_val[2], delta_t = X_val[-4,:], delta_n = X_val[-3,:], y = y_val, plot_result = True)
	plt.title('Validation set, rescaled')

	eval_fit_new_w_constant(best_val_ind, u = X[0], v = X[1], r = X[2], delta_t = X[-2,:], delta_n = X[-4,:], y = y, plot_result = True)
	plt.title('Training set, rescaled')

	print('Best equation rescaled:',eval_fit_new_w_constant(best_val_ind, u = X_val[0], v = X_val[1], r = X_val[2], delta_t = X_val[-4,:], delta_n = X_val[-3,:], y = y_val, return_str = True))





#history
plt.figure()
plt.semilogy(val_acc)
plt.semilogy(train_acc)
plt.grid()
plt.xlabel('Generations')
plt.ylabel('MSE')
plt.legend(['Validation acc', 'Training acc'])
plt.show()


""" div


			#new_ind = gp.PrimitiveTree.from_string(str(exp_str),pset)
			
			# #exp_str = 'a*b*c**2*d**3*sin(a*b)'
			# parts = str(exp_str).split('+')
			
			# for part in parts:
			# 	part2 = part.split('*')
			# 	new_list = []
			# 	cnt = 0
			# 	print(part2)
				
			# 	for i, var in enumerate(part2):

			# 		print(len(var))
					

			# 		if cnt != 0:
			# 			cnt = cnt - 1
			# 			continue

			# 		if i+2 < len(part2):
			# 			if part2[i+1] == '':
			# 				cnt = 2 
			# 				for i in range(int(part2[i+2])):
			# 					new_list.append(var)
			# 			else:
			# 				new_list.append(var)

			# 		# elif i+1 < len(part2):
			# 		# 	if part2[i+1] != '':
			# 		# 		new_list.append(var)
			# 		if i+1 == len(part2):
			# 			new_list.append(var)

					

			# 	print(new_list)
			# 	exit()

				



			# #recreate DEAP string
			
			# split = str(exp_str).split('+')
			# print(split)
			# for part2 in split:
			# 	tot = '' 
			# 	var_list = part2.split('*')
			# 	cnt = 0
			# 	for i, var in enumerate(var_list):
			# 		if cnt != 0:
			# 			cnt = cnt - 1
			# 			continue

			# 		if var == '':
			# 			cnt = 1
			# 			tot = tot + var_list[i-1] + ')'

			# 		elif i+1 == len(var_list):
			# 			tot = tot  +var+ ')'

			# 		else:
			# 			tot = tot + 'mul(' + var + ','
					
			# 	tot = tot + ')'* (tot.count('(') - tot.count(')'))


			# 	print(tot)
				


"""
