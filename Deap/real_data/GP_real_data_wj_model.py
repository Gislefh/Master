"""
Genetic Programming on real data. 

included the water jet model

The waterjet model need work with real data.
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

bag_1 = 'hal_control_2018-12-11-10-53-26_0' #large!
bag_2 = 'hal_control_2018-12-11-11-49-22_0' #similar to bag1 but smaller
bag_3 = 'hal_control_2018-12-11-12-13-58_0' #
bag_4 = 'hal_control_2018-12-11-12-13-58_0'



# bag path
path = '/home/gislehalv/Master/Data/'


bagFile_path_train = path + bag_1 + '.bag'

bagFile_path_val = path + bag_2 + '.bag'


# get data
X = my_lib.open_bag(bagFile_path_train, plot=False, thr_bucket = False, filter_cutoff = 0.025)
#exit()
X_val = my_lib.open_bag(bagFile_path_val, plot=False, thr_bucket = False, filter_cutoff = 0.025)
"""
X = [u, v, r, du, dv, dr, jet_rpm, nozzle_angle, bucket, interp_arr], interp_arr= time. 
Notes:
- nozzle angle is not the angle but in the range[-100, 100], but the ral angle is in the range[-27, 27] deg
- interp_arr is the time variable 
- bucket shuld be > 95 for all data
"""

####--- water jet model ----
def jet_model(nu, jet_rpm, delta_nozzle):
	#constants
	lever_CGtowj_port = [-3.82, -0.475]
	lever_CGtowj_stb = [-3.82, 0.475]
	rpm_slew_rate = [2000, -2000]
	nozzle_slew_rate = [1.3464, -1.3464]
	rpm_min_max = [0, 2000]

	if jet_rpm > rpm_min_max[1]:
		jet_rpm = rpm_min_max[1]
	elif jet_rpm < rpm_min_max[0]:
		jet_rpm = rpm_min_max[0]
	
	if 0:#prev_jet_input and prev_noz_input:
		#rate limiter rpm
		prev_jet_input.append(jet_rpm)
		jet_now = prev_input[-1]
		jet_prev = prev_input[-2]

		rate = (jet_now - jet_prev)/(t_now-t_prev)
		if rate > rpm_slew_rate[0]:
			new_rpm = (t_now-t_prev)*rpm_slew_rate[0] + jet_prev
		elif rate < rpm_slew_rate[1]:
			new_rpm = (t_now-t_prev)*rpm_slew_rate[1] + jet_prev

		#rate limiter nozzle
		prev_noz_input.append(delta_nozzle)
		jet_now = prev_noz_input[-1]
		jet_prev = prev_noz_input[-2]
		rate = (noz_now - noz_prev)/(t_now-t_prev)

		if rate > nozzle_slew_rate[0]:
			new_rpm = (t_now-t_prev)*nozzle_slew_rate[0] + jet_prev
		elif rate < nozzle_slew_rate[1]:
			new_rpm = (t_now-t_prev)*nozzle_slew_rate[1] + jet_prev

	#rpm2thrust
	speed = nu[0] * 1.94384 # knots 
	a0 = 6244.15
	a1 = -178.46
	a2 = 0.881043
	thrust_unscaled = a0 + a1*speed + a2*(speed**2)

	r0 = 85.8316
	r1 = -1.7935
	r2 = 0.00533
	rpm_scale = 1/4530*(r0 + r1*jet_rpm + r2 * (jet_rpm **2))

	thrust = rpm_scale * thrust_unscaled


	#waterjet port
	#force
	Fx = thrust*np.cos(delta_nozzle)
	Fy = thrust*np.sin(delta_nozzle)
	#moment
	Nz_port = (lever_CGtowj_port[0]*Fy)- (lever_CGtowj_port[1]*Fx)
	Nz_stb = (lever_CGtowj_stb[0]*Fy)- (lever_CGtowj_stb[1]*Fx)

	#tau_b_port = [Fx, Fy, Nz_port]
	#tau_b_stb = [Fx, Fy, Nz_stb]

	tau_b =  [2*Fx, 2*Fy, Nz_port + Nz_stb]#np.add(tau_b_port, tau_b_stb)
	#prev_jet_input.append(jet_rpm)
	#prev_noz_input.append(delta_nozzle)
	return tau_b



tau_b = np.zeros((3, np.shape(X)[1]))
for i in range(np.shape(X)[1]):
	nozzle_angle = X[7,i]* (27/100) # to deg
	jet_rpm = X[6,i]
	nu = X[0:3, i]
	tau_b[:,i] = jet_model(nu, jet_rpm, nozzle_angle)

if 0:
	plt.figure()
	plt.subplot(311)
	plt.plot(X[-1], tau_b[0,:]) 
	plt.ylabel('tau_x')
	plt.grid()
	plt.subplot(312)
	plt.plot(X[-1], tau_b[1,:]) 
	plt.ylabel('tau_y')
	plt.grid()
	plt.subplot(313)
	plt.plot(X[-1], tau_b[2,:]) 
	plt.ylabel('tau_z')
	plt.grid()

	plt.figure()
	plt.subplot(211)
	plt.plot(X[-1], new_nozz_ang)
	plt.ylabel('nozzle angle')
	plt.grid()
	plt.subplot(212)
	plt.plot(X[-1], X[6])
	plt.ylabel('jet rpm')
	plt.grid()

	plt.show()
	exit()

tau_b_val = np.zeros((3, np.shape(X_val)[1]))
for i in range(np.shape(X_val)[1]):
	nozzle_angle = X_val[7,i]* (27/100) # to deg
	jet_rpm = X_val[6,i]
	nu = X_val[0:3, i]
	tau_b_val[:,i] = jet_model(nu, jet_rpm, nozzle_angle)

#reconstruct X
X_val = np.concatenate((X_val[0:6], tau_b_val, X_val[-2:]))
X = np.concatenate((X[0:6], tau_b, X[-2:]))




### scaling
#scaling to zero mean
scaling = False
if scaling:
	X_orig = X.copy()
	X_val_orig = X_val.copy()
	for i in range(9):
		X[i] = X[i]  / np.std(X[i])
		X_val[i] = X_val[i] /np.std(X_val[i])




## what variable to use as y
solve_for_du = True 
solve_for_dv = False
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




pset = gp.PrimitiveSet("MAIN", 6)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.abs, 1)
#pset.addPrimitive(np.sin, 1)
#pset.addPrimitive(np.cos, 1)
#pset.addPrimitive(square, 1)

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


#either return_str = True or plot_result = True, not both. 

def eval_fit_new_w_constant(individual, u, v, r, tau_x, tau_y, tau_z, y, return_str = False, plot_result = False):
	#print('individual: ',individual)
	funcs, str_list = split_tree(individual)
	F_list = []
	

	#top root is not 'add'
	if len(funcs) == 1:

		F = funcs[0](u, v, r, tau_x, tau_y, tau_z)
		F_trans = np.transpose(F)

		p = np.dot(np.dot(F_trans,F),np.dot(F_trans,y)) 
		p = [p]


	#top root is 'add'
	else:
		for func in funcs:
			F_list.append(func)
		F = np.zeros((len(y), len(F_list)))

		for i, function in enumerate(F_list):
			F[:,i] = np.squeeze(function(u, v, r, tau_x, tau_y, tau_z))

		F_trans = np.transpose(F)
		try:
			p = np.dot(np.linalg.inv(np.dot(F_trans,F)),np.dot(F_trans,y))  
		except:
			#print('Singular Matrix for: ', individual)
			mse = 1000 # large number
			return(mse,)

	tot_func = np.zeros((len(y)))

	for i, func in enumerate(funcs):
		tot_func = np.add(tot_func, p[i]*func(u, v, r, tau_x, tau_y, tau_z))


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

toolbox.register("evaluate", eval_fit_new_w_constant, u = X[0], v = X[1], r = X[2], tau_x = X[6], tau_y = X[7], tau_z = X[8], y = y, return_str = False)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))


#constants for the GP alg.
pop_size = 10000
mate_prob = 0.5
mut_prob = 0.3
generations = 300

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
best_val = 10000

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

	val_score = eval_fit_new_w_constant(hof[0], u = X_val[0], v = X_val[1], r = X_val[2], tau_x = X_val[6], tau_y = X_val[7], tau_z = X_val[8], y = y_val, return_str = False)[0]
	val_acc.append(val_score)
	print('validation score: ',val_score)
	

	#save best val 
	if  val_score < best_val:
		best_val_ind = hof[0]
		best_val = val_score
		print('Saved as new best')


	#test result on validation set
	if record['min'] < 1e-5:
		mse = eval_fit_new_w_constant(hof[0], u = X_val[0], v = X_val[1], r = X_val[2], tau_x = X_val[6], tau_y = X_val[7], tau_z = X_val[8], y = y_val, return_str = False)
		print('mse for validation: ', mse)
		if mse[0] < 1e-5:
			#print clean eq, and lisp eq
			print('Final result:',eval_fit_new_w_constant(hof[0], u = X_val[0], v = X_val[1], r = X_val[2], tau_x = X_val[6], tau_y = X_val[7], tau_z = X_val[8], y = y_val, return_str = True))
			print(hof[0])

			#plot
			eval_fit_new_w_constant(hof[0], u = X_val[0], v = X_val[1], r = X_val[2], tau_x = X_val[6], tau_y = X_val[7], tau_z = X_val[8], y = y_val, plot_result = True)
			plt.title('Validation set')

			eval_fit_new_w_constant(hof[0], u = X[0], v = X[1], r = X[2], tau_x = X[6], tau_y = X[7], tau_z = X[8], y = y, plot_result = True)
			plt.title('Training set')
			plt.show()
			exit()


# ### scaling
# if scaling:
# 	X = X_orig.copy()
# 	X_val = X_val_orig.copy()
		


#use the best individual w.r.t the validation set 
print('Reached the max number of generations')
print('Best equation:',eval_fit_new_w_constant(best_val_ind, u = X_val[0], v = X_val[1], r = X_val[2], tau_x = X_val[6], tau_y = X_val[7], tau_z = X_val[8], y = y_val, return_str = True))
eval_fit_new_w_constant(best_val_ind, u = X_val[0], v = X_val[1], r = X_val[2], tau_x = X_val[6], tau_y = X_val[7], tau_z = X_val[8], y = y_val, plot_result = True)
plt.title('Validation set')

eval_fit_new_w_constant(best_val_ind, u = X[0], v = X[1], r = X[2], tau_x = X[6], tau_y = X[7], tau_z = X[8], y = y, plot_result = True)
plt.title('Training set')


#rescale back 
if scaling:
	X = X_orig.copy()
	X_val = X_val_orig.copy()

	eval_fit_new_w_constant(best_val_ind, u = X_val[0], v = X_val[1], r = X_val[2], tau_x = X_val[6], tau_y = X_val[7], tau_z = X_val[8], y = y_val, plot_result = True)
	plt.title('Validation set, rescaled')

	eval_fit_new_w_constant(best_val_ind, u = X[0], v = X[1], r = X[2], tau_x = X[6], tau_y = X[7], tau_z = X[8], y = y, plot_result = True)
	plt.title('Training set, rescaled')



plt.show()