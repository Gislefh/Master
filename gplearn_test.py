import numpy as np
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from sklearn import preprocessing
import pydotplus
from PIL import Image
import io


##  -- import data -- 
file_path = '/home/gislehalv/Master/Data/CSV Data From Bags/hal_control_2018-12-11-12-19-11_0.csv'
data = np.loadtxt(file_path, delimiter = ',')
## data = [u, v, r, du, dv, dr, engine rpm, steering nozzle, bucket, time]

#data = data[:,0:5000]


#X = np.concatenate((data[0:3],data[6:8]))  #[u,v,r,thro,noz]
X = np.concatenate((data[6:7],data[0:1]))
Y = data[3, :].reshape(-1,1) #du
t = data[-1, :]
print(np.shape(Y))

# plt.figure()
# plt.plot(np.expand_dims(t,axis=1),np.transpose(X))
# plt.legend(['u','v','r','thro','noz'])
# plt.figure()
# plt.plot(t,Y)

#--- preeprosess
#sklearn 
# scalerX = preprocessing.MinMaxScaler().fit(X)
# X = scalerX.transform(X) 
# scalerY = preprocessing.StandardScaler().fit(Y)
# Y = scalerY.transform(Y)

#numpy
for i in range(np.shape(X)[0]):
	X[i, :] = X[i, :] -np.mean(X[i, :])
	X[i, :] = X[i, :] / np.std(X[i, :])
Y = Y - np.mean(Y)
Y = Y / np.std(Y)




# plt.figure()
# plt.plot(np.expand_dims(t,axis=1),np.transpose(X))
# plt.legend(['u','v','r','thro','noz'])
# plt.figure()
# plt.plot(Y)
# plt.show()


est_gp = SymbolicRegressor(population_size=4000, 
							generations=30, 
							tournament_size=20, 
							stopping_criteria=0.0, 
							const_range=(-1.0, 1.0),
							init_depth=(2, 6), 
							init_method='half and half', 
							function_set=('add','mul', 'sub', 'div', 'abs'),#, 'sqrt', 'sin'), 
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

est_gp.fit(np.transpose(X), Y)

print(est_gp._program)

Y_est = est_gp.predict(np.transpose(X))

#tree
graph = pydotplus.graphviz.graph_from_dot_data(est_gp._program.export_graphviz())
img = Image.open(io.BytesIO(graph.create_png()))
tree = np.asarray(img)
plt.figure()
plt.imshow(tree)


#result
print(est_gp.score(np.transpose(X),Y))
plt.figure()
plt.plot(t,Y)
plt.plot(t,Y_est)
plt.legend(['true','estimated'])
plt.show()


