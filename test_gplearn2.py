import numpy as np
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
import pydotplus
from PIL import Image
import io
from sklearn import preprocessing

x0 = np.random.rand(100,1)
x1 = np.random.rand(100,1)

y = np.zeros((len(x0),1))
for i in range(len(x0)):
	y[i,0] = x0[i]*x1[i] + np.sin(x0[i]*x0[i]) - x0[i]/x1[i] + 1
	
x0 = x0.reshape(-1,1)
x1 = x1.reshape(-1,1)
x = np.concatenate((x0,x1),axis = 1)
#y = y.reshape(-1,1)

print(np.shape(x), np.shape(y))




# scalerX = preprocessing.StandardScaler().fit(x)
# x = scalerX.transform(x) 

# scalerY = preprocessing.StandardScaler().fit(y)
# y = scalerY.transform(y)


est_gp = SymbolicRegressor(population_size=4000, 
							generations=2, 
							tournament_size=20, 
							stopping_criteria=0.0001, 
							const_range=(-1.0, 1.0),
							init_depth=(2, 6), 
							init_method='half and half', 
							function_set=('add','mul', 'sub', 'mul', 'div', 'sqrt', 'sin'), 
							metric='mse', 
							parsimony_coefficient=0.001, 
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
print(np.shape(tree))
print(type(graph))
exit()
plt.figure()
plt.imshow(tree)
plt.figure()
plt.plot(list(range(np.shape(x)[0])),y)
plt.plot(list(range(np.shape(x)[0])),Y_est)
plt.legend(['true','estimated'])
plt.show()




