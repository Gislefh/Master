#My Lib
import numpy as np
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor

from scipy.integrate import solve_ivp, cumtrapz
import pydotplus
from PIL import Image
import io
from scipy import signal


#plots the tree structure
def show_result(est_gp, X, Y, t, plot_show = False):
	graph = pydotplus.graphviz.graph_from_dot_data(est_gp._program.export_graphviz())
	img = Image.open(io.BytesIO(graph.create_png()))
	tree = np.asarray(img)
	plt.figure()
	plt.imshow(tree)

	print('RESULTING EQ: ',est_gp._program)
	Y_est = est_gp.predict(X)

	print('r²: ',est_gp.score(X,Y))
	plt.figure()
	plt.plot(t,Y)
	plt.plot(t,Y_est)
	plt.legend(['data', 'pred'])

	if plot_show:
		plt.show()


"""
_IN_
time: 	[t_start, t_stop, t_step] -> start time, stop time, step length
mdk:	[m, d, k] -> mass, damping, spring stiffness
tau:	input to the system -> either: 'sin', 'step' or square


_OUT_
t,ddx,dx,x,inp: sim time, d²x/d²t, dx/dt, x, input to the system

"""
def MSD(time = [0,10,0.1], mdk = [1,1,1], x0 = 0, dx0 = 0, tau = 'step'): #Mass spring damper system
	
	#----Inputs
	def inp_step(t): 
		if t < 5:
			return 0
		#elif (t > 5) and (t < 10):
		#	return 10
		else:
			return 10

	def inp_sin(t):
		return np.sin(t)

	def inp_square(t):
		return signal.square(t/3)

	inp = []


	#----SYS
	#const
	m,d,k = mdk
	#print('true eq: ',1/m,'*tau -',d/m,'*dx -',k/m,'*x')

	#time
	t_start, t_stop, t_step = time
	t = list(np.arange(t_start,t_stop,t_step))

	#matrices
	A = np.array([[0, 1],[-k/m, -d/m]])
	B = np.array([[0], [1/m]])


	if tau == 'step':
		def sys(t,X):
			dX = np.dot(A,X) + np.dot(B,inp_step(t))
			return dX

	elif tau == 'sin':
		def sys(t,X):
			dX = np.dot(A,X) + np.dot(B,inp_sin(t))
			return dX

	elif tau == 'square':
		def sys(t,X):
			dX = np.dot(A,X) + np.dot(B,inp_square(t))
			return dX

	else:
		print('not a valid input for < tau >')
		exit()

	sol = solve_ivp(sys,[t[0], t[-1]],[x0,dx0], vectorized = True,  max_step = t_step)# t_eval = t,
	#acc = np.diff(sol.y[1,:])/np.diff(sol.t)
	
	#input
	for i in range(len(sol.t)):
		if tau == 'step': 
			inp.append(inp_step(sol.t[i]))
		elif tau == 'sin': 
			inp.append(inp_sin(sol.t[i]))
		elif tau == 'square': 
			inp.append(inp_square(sol.t[i]))


	acc = np.multiply((1/m),inp) - np.multiply((d/m),sol.y[1, :]) - np.multiply((k/m),sol.y[0, :])
	inp = np.array(inp).reshape(-1,1)


	return  np.array(sol.t), np.array(acc).reshape(-1,1), np.array(sol.y[1, :]).reshape(-1,1), np.array(sol.y[0, :]).reshape(-1,1), inp



#def Preprocess