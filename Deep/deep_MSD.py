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

import pygraphviz as pgv
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/home/gislehalv/Master/scripts')
import my_lib


## Simulation
time = [0,300,0.1]
mdk = [2, 1, 3]

t,ddx,dx,x,tau = my_lib.MSD(time = time, mdk = mdk, tau = 'square')


#Add Noise


#Add Time Delay


#Scaling / Preprocessing


##Functions
def add3(x,y,z):
	return x+y+z

def mul3(x,y,z):
	return x*y*z


pset = gp.PrimitiveSet("MAIN", 3)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(add3, 3)
#pset.addPrimitive(mul3, 3)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(operator.abs, 1)
pset.addEphemeralConstant("rand101", lambda: np.random.uniform(-1,1))

#Variable names 
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


def eval_fit(individual, dx, x, tau, ddx):
	func = toolbox.compile(expr=individual)

	#Mean squared error
	mse = ((func(dx,x,tau) - ddx )**2)
	return (math.fsum(mse),)


toolbox.register("evaluate", eval_fit,  dx = dx, x = x, tau = tau, ddx = ddx)

toolbox.register("select", tools.selTournament, tournsize=3)
#toolbox.register("select", tools.selDoubleTournament, parsimony_size = 1.5,fitness_first = True, fitness_size = 5)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.register("mutate", gp.mutEphemeral, mode = 'all') #mode all: changes the value of all the ephemeral constants
                                                        #mode one: changes the value of one of the individual ephemeral constants

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))

def main():
    random.seed(316)

    pop = toolbox.population(n=3000)
    hof = tools.HallOfFame(1)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 20, stats=mstats,
                                   halloffame=hof, verbose=True)


    # ### Graphviz Tree ###
    expr = toolbox.individual()
    nodes, edges, labels = gp.graph(hof[0])
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw("tree.pdf")

    #sympify
    locals = {
    'mul': lambda x, y : x * y,
    'add': lambda x, y : x + y,
    'add3': lambda x, y, z: x+y+z,
    'sub': lambda x, y : x - y,
    'protectedDiv': lambda x, y: x / y,
    'neg': lambda x: -x,
    'sin': lambda x: sin(x),
    'cos': lambda x: cos(x)
    }

    hof_str = str(hof[0])
    eq = sympify(hof_str,locals = locals)
    print('Clean equation: ', eq)
    print('LISP:', hof_str)
    print(type(eq))


    # Plot
    func = toolbox.compile(expr=hof[0])

    plt.figure()
    plt.plot(t,func(dx,x,tau))
    plt.plot(t,ddx)
    plt.legend(['prediction','actual equation'])


    # plt.figure()
    # plt.subplot(311)
    # plt.plot(t,dx)
    # plt.legend(['dx'])

    # plt.subplot(312)
    # plt.plot(t,x)
    # plt.legend(['x'])

    # plt.subplot(313)
    # plt.plot(t,tau)
    # plt.legend(['tau'])
    plt.show()

    

    return None



if __name__ == "__main__":
    main()