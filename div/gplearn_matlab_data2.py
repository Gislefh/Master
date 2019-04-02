import numpy as np
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from sklearn import preprocessing
import pydotplus
from PIL import Image
import io


##  -- import data -- 
file_path = '/home/gislehalv/Master/Data/data_fom_MatlabSim/du_data8.csv'
data = np.loadtxt(file_path, delimiter = ',')
## data = [u,v,r,U,du,dv,dr,dU,force_x,force_y,force_z,time]


X = np.concatenate((data[:, 0:3],data[:, 8:11]),axis = 1)  #[u,v,r,fx,fy,fz]
Y = data[:, 4:7] #du,dv,dr
t = data[:, -1]

plt.figure()
plt.plot(np.expand_dims(t,axis=1),X)
plt.legend(['u','v','r','thro','noz'])

#--- preeprosess
for i in range(np.shape(X)[1]):
	if np.std(X[:, i]):
		X[:, i] = X[:, i] -np.mean(X[:, i])
		X[:, i] = X[:, i] / np.std(X[:, i])

for i in range(np.shape(Y)):
	Y[:, i] = Y[:, i] - np.mean(Y[:, i])
	Y[:, i] = Y[:, i] / np.std(Y[:, i])


plt.figure()
plt.plot(np.expand_dims(t,axis=1),X)
plt.legend(['u','v','r','fx','fy','fz'])

est_gp = SymbolicRegressor(population_size=8000, 
							generations=3, 
							tournament_size=30, 
							stopping_criteria=0.0, 
							const_range=(-1.0, 1.0),
							init_depth=(2, 6), 
							init_method='half and half', 
							function_set=('add','mul', 'sub', 'abs'),#, 'sqrt', 'sin','div') 
							metric='mean absolute error', 
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
print('DU')
du_pred = est_gp.fit(X, Y[:,0])
print('DV')
dv_pred = est_gp.fit(X, Y[:,0])
print('DR')
dr_pred = est_gp.fit(X, Y[:,0])
est_gp.fit(X, Y)







#tree
graph = pydotplus.graphviz.graph_from_dot_data(est_gp._program.export_graphviz())
img = Image.open(io.BytesIO(graph.create_png()))
tree = np.asarray(img)
plt.figure()
plt.imshow(tree)

#result
Y_est = est_gp.predict(X)
plt.figure()
plt.plot(t,Y)
plt.plot(t,Y_est)
plt.legend(['data', 'pred'])
plt.show()




