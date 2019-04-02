import numpy as np
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from sklearn import preprocessing
import pydotplus
from PIL import Image
import io


##  -- import data -- 
file_path = '/home/gislehalv/Master/Data/data_fom_MatlabSim/du_data9.csv'
data = np.loadtxt(file_path, delimiter = ',')
## data = [u,v,r,U,du,dv,dr,dU,force_x,force_y,force_z,time]


X = np.concatenate((data[:, 0:3],data[:, 8:11]),axis = 1)  #[u,v,r,fx,fy,fz]

#testing with fx,fy,fz =/ 1000
# X[:, 3] = X[:, 3]/100
# X[:, 4] = X[:, 4]/100
# X[:, 5] = X[:, 5]/100




Y = data[:, 4] # du = 4, dv = 5, dr = 6
t = data[:, -1]

# test fewer inputs
#X = np.concatenate((data[:,0].reshape(-1,1),data[:,8].reshape(-1,1)), axis = 1) #[u,fx]
X = np.concatenate((data[:, 0:3],data[:, 8].reshape(-1,1)),axis = 1)  #[u,v,r,fx]
# X[:, 3] = X[:, 3]/100

#--- preeprosess
plt.figure()
plt.plot(np.expand_dims(t,axis=1),X)
plt.legend(['u','v','r','thro','noz'])

std_x = []
mean_y = []
for i in range(np.shape(X)[1]):
	if np.std(X[:, i]):
	#X[:, i] = X[:, i] - np.mean(X[:, i])
		X[:, i] = X[:, i] / np.std(X[:, i])

#Y = Y - np.mean(Y)
Y = Y / np.std(Y)



plt.figure()
plt.plot(np.expand_dims(t,axis=1),X)
plt.legend(['u','v','r','fx','fy','fz'])


X = X[:4000]
Y = Y[:4000]
t = t[:4000]


est_gp = SymbolicRegressor(population_size=8000, 
							generations=20, 
							tournament_size=5, 
							stopping_criteria=0.0, 
							const_range=(-3, 3),
							init_depth=(2, 6), 
							init_method='half and half', 
							function_set=('add','mul', 'sub', 'abs'),#, 'sqrt', 'sin','div') 
							metric='mse', 
							parsimony_coefficient=0.01, 
							p_crossover=0.80, 
							p_subtree_mutation=0.01, 
							p_hoist_mutation=0.1, 
							p_point_mutation=0.01, 
							p_point_replace=0.05, 
							max_samples=1.0, 
							warm_start=False, 
							n_jobs=1, 
							verbose=1, 
							random_state=1)


est_gp.fit(X, Y)

#tree
graph = pydotplus.graphviz.graph_from_dot_data(est_gp._program.export_graphviz())
img = Image.open(io.BytesIO(graph.create_png()))
tree = np.asarray(img)
plt.figure()
plt.imshow(tree)

print('RESULTING EQ: ',est_gp._program)




Y_est = est_gp.predict(X)

#result
print('r²: ',est_gp.score(X,Y))

plt.figure()
plt.plot(t,Y)
plt.plot(t,Y_est)
plt.legend(['data', 'pred'])







# --- integrate
dt = t[1]-t[0]
print(dt)
y_int = np.zeros((len(Y_est)))
print(np.shape(y_int))
y_int[0] = np.trapz(Y_est[0:2],dx = dt)

for i in np.arange(1,len(Y_est)):
	dt = t[i]-t[i-1]
	y_int[i] = np.trapz(Y_est[i-1:i+1], dx = dt) + y_int[i-1]


print(np.shape(y_int))


# plt.figure()
# plt.plot(y_int)
# plt.plot(data[5, :])
plt.show()




