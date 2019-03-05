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


pset = gp.PrimitiveSet("MAIN", 3)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(operator.abs, 1)
pset.addPrimitive(np.sin, 1)
#pset.addEphemeralConstant("randUn", lambda: random.randint(-1,1))

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



MSE_pars = False
MSE = False
LSR = True


def eval_fit(individual, ddx, dx, x, tau):
	func = toolbox.compile(expr=individual)

	if MSE:
		
		mse = ((func(x) - y )**2)
		return (math.fsum(mse)/len(y),)

	if MSE_pars:
		mse = math.fsum(((func(x) - y )**2))/len(y)

		mse_p = mse + (pc *len(individual))

		return (mse_p,)

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


toolbox.register("evaluate", eval_fit, ddx = ddx, dx = dx, x = x, tau = tau)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=4))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=4))

### main algorithm ##
#constants

pop_size = 500
mate_prob = 0.2
mut_prob = 0.6
generations = 20

#parsimony coefficient
if MSE_pars:
	pc = 0.2

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


func = toolbox.compile(expr=hof[0])	
new_str, sol = my_lib.new_string_from_LSR(func, hof[0], ddx, dx, x, tau)
	
print(new_str)