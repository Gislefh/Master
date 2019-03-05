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
sim_time = [0, 500, 0.1]

###   INPUT   ###
def inp_step_x(t,states):
	if t < 5:
		return [[0],[0],[0]]
	else:
		return [[4000],[0],[0]]

def inp_step_y(t,states):
	if t < 5:
		return [[0],[0],[0]]
	else:
		return [[0],[4000],[0]]

def inp_step_yz(t,states):
	if t < 5:
		return [[0],[0],[0]]
	else:
		return [[0],[4000],[4000]]

def inp_test(t,states):
	if states[3] < 10:
		return [[400000],[0],[0]]
	else:
		return [[0],[0],[0]]

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


d2r = np.pi/180
r2d = 180/np.pi
prev_input = []
def inp_zig_zag(t,states):

	if not prev_input or t < 5:

		out = [[0],[0],[0]]
		prev_input.append(out)
		start = True
		return out

	if prev_input[-1] == [[0],[0],[0]]:
		out = [[4000],[30],[500]]
		prev_input.append(out)

		return out

	if states[3] < 4: #surge speed
		fx = 400
	else:
		fx = 0

	if states[2] < np.abs(d2r * 20): #yaw
		fy = prev_input[-1][1][0]
		fz = prev_input[-1][2][0]

	elif states[2] > d2r * 20:
		fy = 30
		fz = 500

	elif states[2] < d2r * 20:
		fy = -30
		fz = -500

	out = [[fx],[fy],[fz]]
	prev_input.append(out)

	return out

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

	# elif t > 500 and t < 600:
	# 	if states[3] < 5: #surge speed
	# 		fx = 8000
	# 	else:
	# 		fx = 3000
	# 	return [[fx],[0],[0]]


	else:
		return [[0],[0],[0]]



def inp_(t,states):
	if states[0]< 3:
		fx = 5000
	else:
		fx = 2000

	if t < 400:

		if np.abs(states[4]) > 1:
			fy = 0
		else:
			fy = 1000 + 2*t

		if np.abs(states[5]) > 1:
			fz = 0
		else:
			fz = 1000 - 2 * t

	else:
		fx = 0
		if np.abs(states[4]) > 1:
			fy = 0
		else:
			fy = 0

		if np.abs(states[5]) > 1:
			fz = 0
		else:
			fz = 0

	out = [[fx],[fy],[fz]]
	return out

prev_states = []
prev_inputs = []
"""
def inp_2(t,states):

	prev_states.append(states)

	if len(prev_states) < 10:
		out = [[5000],[0],[0]]
		prev_inputs.append(out)
		return out

	if np.abs(prev_states[-1][3][0] - prev_states[-2][3][0]) < 0.00005:
		fx = prev_inputs[-1][0][0] + 500
	else:

		fx = prev_inputs[-1][0][0]
	print(fx)
	out = [[fx],[0],[0]]
	prev_inputs.append(out)
	return out
"""

def inp_3(t,states):
	if t < 200:
		fx = signal.square(t/6) * 2000 + 2000
		fy = signal.square(t/4) * 700 - 700
		fz = signal.square(t/10) * 1000 + 1000

	elif t < 300:
		fx = np.sin(t/6) * 1000 + 1000
		fy = np.sin(t/4) * 1100 - 400
		fz = np.sin(t/10) * 800 + 1000

	elif t < 400:
		fx = 5000
		fy = 0
		fz = 0

	else:
		fx = 0
		fy = 0
		fz = 0

	return [[fx],[fy],[fz]]



### RUN SIM  ###
tictoc = TicToc()
tictoc.tic()
X = my_lib.boat_simulation(inp_3, time = sim_time)
tictoc.toc()
my_lib.boat_sim_plot(X, show = False)


###TEST###
if 0:
	du = []
	dv = []
	dr = []

	for i in range(len(X[-1])):
		du.append( 2.026e-4*X[6,i] + X[1,i]*X[2,i] - 0.0101*X[0,i] -0.0492*np.abs(X[0,i])*X[0,i] )
		dv.append( 2.026e-4*X[7,i] + X[0,i]*X[2,i] - 0.045*X[1,i] - 0.4052*np.abs(X[1,i])*X[1,i] )
		dr.append( 4.78e-5*X[8,i] -0.2595*(X[1,i]+X[2,i]) - 0.6027*np.abs(X[2,i])*X[2,i] )

	plt.figure()
	plt.plot(X[-1],du)
	plt.plot(X[-1],X[3])
	plt.figure()
	plt.plot(X[-1],dv)
	plt.plot(X[-1],X[4])
	plt.figure()
	plt.plot(X[-1],dr)
	plt.plot(X[-1],X[5])


	#plt.legend(['eq', 'sim'])
	plt.show()
	exit()

# #######


# Operators
pset = gp.PrimitiveSet("MAIN", 6)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(operator.abs, 1)
#pset.addEphemeralConstant("randUn", lambda: random.uniform(-1,1))

#Variable names 
pset.renameArguments(ARG0='u')
pset.renameArguments(ARG1='v')
pset.renameArguments(ARG2='r')
pset.renameArguments(ARG3='fx')
pset.renameArguments(ARG4='fy')
pset.renameArguments(ARG5='fz')


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

def eval_fit(individual,  u, v, r, fx, fy, fz, du):
	func = toolbox.compile(expr=individual)

	# #Mean squared error - seems to give the better results than rmse and mae 
	# if MSE:
	# 	mse = ((func(dx,x,tau) - ddx )**2)
	# 	return (math.fsum(mse)/len(x),)

	# #Root mean squared error
	# if RMSE:
	# 	rmse = np.sqrt(math.fsum((func(dx,x,tau) - ddx )**2)/len(tau))
	# 	return (rmse,)

	# #Mean absolute error
	# if MAE:
	# 	mae = math.fsum(np.abs(func(dx,x,tau) - ddx))/len(tau)
	# 	return (mae,)

	#Least Squares Regression
	if LSR:
		def fun(X):
			function = func(X[0]*u, X[1]*v, X[2]*r, X[3]*fx, X[4]*fy, X[5]*fz)
			se = (du-function)**2
			out = np.squeeze(se)

			return out

		x0 = np.array([1, 1, 1, 1e-3, 1e-3, 1e-3])
		sol = optimize.least_squares(fun,x0)

		mse = math.fsum((du - func(sol.x[0]*u, sol.x[1]*v, sol.x[2]*r, sol.x[3]*fx, sol.x[4]*fy, sol.x[5]*fz))**2)/len(u)
		return (mse,)


toolbox.register("evaluate", eval_fit,  u = X[0], v = X[1], r = X[2], fx = X[6], fy = X[7], fz = X[8], du = X[3])

toolbox.register("select", tools.selTournament, tournsize=20)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


#shrinks the individual by chosing randomly a branch and replacing it with one of the branch’s arguments
#not working well :(
#toolbox.register("mutate", gp.mutShrink, ) 


#comment out for no constant terms
#toolbox.register("mutate", gp.mutEphemeral, mode = 'all') #mode all: changes the value of all the ephemeral constants
                                                        #mode one: changes the value of one of the individual ephemeral constants
                                                        

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=4))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=4))

def main():
	#random.seed(200)
	pop = toolbox.population(n=5000)
	hof = tools.HallOfFame(1)

	stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
	stats_size = tools.Statistics(len)
	mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
	mstats.register("avg", np.mean)
	mstats.register("std", np.std)
	mstats.register("min", np.min)
	mstats.register("max", np.max)

	## Fit ##
	pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.3, 100, stats=mstats,
								halloffame=hof, verbose=True)

	func = toolbox.compile(expr=hof[0])

	### find the weights and add them to the string ###
	if LSR:
		#find good constants
		def fun(S):
			function = func(S[0]*X[0], S[1]*X[1], S[2]*X[2], S[3]*X[6], S[4]*X[7], S[5]*X[8])
			se = (X[3]-function)**2
			out = np.squeeze(se)
			return out

		x0 = np.array([1,1,1,1e-3,1e-3,1e-3])
		sol = optimize.least_squares(fun,x0)

		#Add the constants to the equation (new_str)
		tmp = str(hof[0])
		new_str = ''
		skip = 0
		for i in range(len(tmp)):
			if tmp[i] == 'u' and tmp[i+1] != 'l' and tmp[i+1] != 'b' :
				new_str = new_str +"mul(u,{:.7f})".format(sol.x[0])
				skip = 1

			elif tmp[i] == 'v':
				new_str = new_str +"mul(v,{:.7f})".format(sol.x[1])
				skip = 1

			elif tmp[i] == 'r':
				new_str = new_str +"mul(r,{:.7f})".format(sol.x[2])
				skip = 1

			elif tmp[i] == 'f':
				if tmp[i+1] == 'x':
					new_str = new_str +"mul(fx,{:.7f})".format(sol.x[3])
				elif tmp[i+1] == 'y':
					new_str = new_str +"mul(fy,{:.7f})".format(sol.x[4])
				elif tmp[i+1] == 'z':
					new_str = new_str +"mul(fz,{:.7f})".format(sol.x[5])
				skip = 2

			elif skip == 0:
				new_str = new_str + tmp[i]

			if skip != 0:
				skip = skip - 1

	# ### Graphviz Tree ###
	if 1:
		
		expr = toolbox.individual()
		nodes, edges, labels = gp.graph(hof[0])
		#Add input weights to the tree
		if LSR:
			for i in nodes:
				if isinstance(labels[i], str):
					if labels[i] == 'u':
						labels[i] = "{:.3f}".format(sol.x[0])+'*u'
					if labels[i] == 'v':
						labels[i] = "{:.3f}".format(sol.x[1])+'*v'
					if labels[i] == 'r':
						labels[i] = "{:.3f}".format(sol.x[2])+'*r'
					if labels[i] == 'fx':
						labels[i] = "{:.3f}".format(sol.x[3])+'*fx'
					if labels[i] == 'fy':
						labels[i] = "{:.3f}".format(sol.x[4])+'*fy'
					if labels[i] == 'fx':
						labels[i] = "{:.3f}".format(sol.x[5])+'*fz'


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
	if 1:
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
		print(eq)
		print(new_str)

	### History Plot ##
	if 1:
		gen = log.select("gen")
		fit_mins = log.chapters["fitness"].select("min")
		size_avgs = log.chapters["size"].select("avg")


		fig, ax1 = plt.subplots()
		line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
		ax1.set_xlabel("Generation")
		ax1.set_ylabel("Fitness", color="b")
		for tl in ax1.get_yticklabels():
			tl.set_color("b")

		ax2 = ax1.twinx()
		line2 = ax2.plot(gen, size_avgs, "r-", label="Average Size")
		ax2.set_ylabel("Size", color="r")
		for tl in ax2.get_yticklabels():
			tl.set_color("r")

		lns = line1 + line2
		labs = [l.get_label() for l in lns]
		ax1.legend(lns, labs, loc="center right")

	# #####  Plot Solution#####
	if 1:
		plt.figure()
		#plt.title('MSE = '+'{:.4f}'.format(mse_result*1e6) + '1e-6')
		if LSR:
			plt.plot(X[-1], func(sol.x[0]*X[0], sol.x[1]*X[1], sol.x[2]*X[2], sol.x[3]*X[6], sol.x[4]*X[7], sol.x[5]*X[8]))
		else:
			plt.plot(X[-1],func(X[0],X[1],X[2],X[6],X[7],X[8]))
		plt.plot(X[-1],X[3])
		plt.xlabel('Time [s]')
		plt.ylabel('du [m/s²]')
		plt.grid()
		plt.legend(['prediction','actual equation'])

		plt.show()

	return None
main()