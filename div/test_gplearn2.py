import numpy as np
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
import pydotplus
from PIL import Image
import io
from sklearn import preprocessing

x = np.random.rand(100,1)*10

#y = -0.0101*x + 0.0492*np.abs(x)*x
y = x + x*x + 10 #np.abs(x)*x + np.sqrt(np.sin(x))
	

# x = x - np.mean(x)
# x = x/np.std(x)
# y = y -np.mean(y)
# y = y/np.std(y)

est_gp = SymbolicRegressor(population_size=8000, 
							generations=30, 
							tournament_size=5, 
							stopping_criteria=0.00001, 
							const_range=(9, 11),
							init_depth=(2, 6), 
							init_method='half and half', 
							function_set= ('add','mul'), #, 'mul', 'abs', 'sub', 'sin', 'sqrt'),    #('add','mul', 'sub', 'div', 'sqrt', 'sin', 'abs'), 
							metric='mse', 
							parsimony_coefficient=0.005, 
							p_crossover=0.9, 
							p_subtree_mutation=0.01, 
							p_hoist_mutation=0.01, 
							p_point_mutation=0.01, 
							p_point_replace=0.05, 
							max_samples=1.0, 
							warm_start=False, 
							n_jobs=1, 
							verbose=1, 
							random_state=None)


est_gp.fit(x, y)

#x = scalerX.inverse_transform(x)
#y = scalerY.inverse_transform(y)


print(est_gp._program)

Y_est = est_gp.predict(x)
#print(est_gp.get_params(True))


print(est_gp.score(x,y))

graph = pydotplus.graphviz.graph_from_dot_data(est_gp._program.export_graphviz())
img = Image.open(io.BytesIO(graph.create_png()))
tree = np.asarray(img)

plt.figure()
plt.imshow(tree)
plt.figure()
plt.plot(list(range(np.shape(x)[0])),y)
plt.plot(list(range(np.shape(x)[0])),Y_est)
plt.legend(['true','estimated'])
plt.show()




