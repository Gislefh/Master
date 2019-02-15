#    This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

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

# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(operator.abs, 1)
pset.addPrimitive(operator.pow, 2)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-5,5))
pset.renameArguments(ARG0='x')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    sqerrors = ((func(x)  - x**2 - 5*x*np.abs(x))**2 for x in points)
    return math.fsum(sqerrors) / len(points)

toolbox.register("evaluate", evalSymbReg, points=[x/1. for x in range(-10,10)])
#toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("select", tools.selDoubleTournament, parsimony_size = 1.5,fitness_first = True, fitness_size = 5)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))

def main():
    #random.seed(316)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 1000, stats=mstats,
                                   halloffame=hof, verbose=True)

    ##sympify
    locals = {
    'mul': lambda x, y : x * y,
    'add': lambda x, y : x + y,
    'sub': lambda x, y : x - y,
    'protectedDiv': lambda x, y: x / y,
    'neg': lambda x: -x,
    'sin': lambda x: sin(x),
    'cos': lambda x: cos(x),
    'abs': lambda x: np.abs(x)
    }

    hof_str = str(hof[0])
    eq = sympify(hof_str,locals = locals)
    print('Clean equation: ', eq)
    print('LISP:', hof_str)
    print(type(eq))



    # PLOT
    func = toolbox.compile(expr=hof[0])
    x = np.arange(0,10,1)
    pred_func = np.zeros((len(x)))
    for cnt, i in enumerate(x):
        pred_func[cnt] = func(i)

    acc_func =  x**2 + 5*x*np.abs(x)
    plt.figure()
    plt.plot(x,pred_func)
    plt.plot(x,acc_func)
    plt.legend(['prediction','actual equation'])
    




    # ### Graphviz Section ###
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


    #plot gen
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

    plt.show()

    return pop, log, hof

if __name__ == "__main__":
    main()
