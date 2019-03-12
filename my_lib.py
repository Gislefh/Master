#My Lib
import numpy as np
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor

from scipy.integrate import solve_ivp, cumtrapz
import pydotplus
from PIL import Image
import io
from scipy import signal
from scipy import optimize

from sympy import sympify, cos, sin

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

"""-------- MASS SPRING DAMPER ---------
_IN_
time: 	[t_start, t_stop, t_step] -> start time, stop time, step length
mdk:	[m, d, k] -> mass, damping, spring stiffness
tau:	input to the system -> either: 'sin', 'step' or square


_OUT_
t,ddx,dx,x,inp: sim time, d²x/d²t, dx/dt, x, input to the system

"""
def MSD(time = [0,10,0.1], mdk = [1,1,1], x0 = 0, dx0 = 0, tau = 'step', time_delay = 0, true_eq = False): #Mass spring damper system
	

	#----Inputs
	def inp_step(t):
		if t <= time_delay:
			t = 0
		else:
			t = t - time_delay
		if t <= 5:
			return 0
		#elif (t > 5) and (t < 10):
		#	return 10
		else:
			return 10

	def inp_sin(t):
		if t < time_delay:
			t = 0
		else:
			t = t - time_delay

		return np.sin(t)

	def inp_square(t):
		if t <= time_delay:
			t = 0
			return 0
		else:
			t = t - time_delay
		return signal.square(t/3)

	inp = []


	#----SYS
	#const
	m,d,k = mdk
	if true_eq:
		print('true eq: ',1/m,'*tau -',d/m,'*dx -',k/m,'*x')

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

	sol = solve_ivp(sys,[t[0], t[-1]],[x0,dx0], vectorized = True,  max_step = t_step, t_eval = t)

	#delayed input

	for i in range(len(sol.t)):
		if tau == 'step': 
			inp.append(inp_step(sol.t[i]))
		elif tau == 'sin': 
			inp.append(inp_sin(sol.t[i]))
		elif tau == 'square': 
			inp.append(inp_square(sol.t[i]))



	#ddx
	acc = np.multiply((1/m),inp) - np.multiply((d/m),sol.y[1, :]) - np.multiply((k/m),sol.y[0, :])
	

	#input
	inp = []
	time_delay = 0
	for i in range(len(sol.t)):
		if tau == 'step': 
			inp.append(inp_step(sol.t[i]))
		elif tau == 'sin': 
			inp.append(inp_sin(sol.t[i]))
		elif tau == 'square': 
			inp.append(inp_square(sol.t[i]))

	inp = np.array(inp).reshape(-1,1)

	return  np.array(sol.t), np.array(acc).reshape(-1,1), np.array(sol.y[1, :]).reshape(-1,1), np.array(sol.y[0, :]).reshape(-1,1), inp


"""
 ------------------ FIND THE CONSTANTS FROM LSR AND CREATE NEW STRING - for MSD sys ----------------
_IN_ 
func: 				the function with arguments (dx, x, tau)
hof:				string to append the weights to
ddx, dx, x, tau: 	the variables

_OUT_ 
new_string: the new string
sol: 		weights from the inputs
"""
def new_string_from_LSR(func, hof, ddx, dx, x, tau):

	#find good constants
	def fun(X):
		se = (ddx - func(X[0]*dx, X[1]*x, X[2]*tau))**2
		se_1d = np.squeeze(se)
		return se_1d
	x0 = np.array([1,1,1])
	sol = optimize.least_squares(fun,x0)

	#Add the constants to the equation (new_str)
	tmp = str(hof)
	new_str = ''
	skip = 0
	for i in range(len(tmp)):
		if tmp[i] == 'd' and tmp[i+1] == 'x':
			new_str = new_str +"mul(dx,{:.7f})".format(sol.x[0])
			skip = 2

		elif tmp[i] == 'x' and tmp[i-1] != 'd':
			new_str = new_str +"mul(x,{:.7f})".format(sol.x[1])
			skip = 1

		elif tmp[i] == 't' and tmp[i+1] == 'a' and tmp[i+2] == 'u':
			new_str = new_str +"mul(tau,{:.7f})".format(sol.x[2])
			skip = 3

		elif skip == 0:
			new_str = new_str + tmp[i]

		if skip != 0:
			skip = skip - 1


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
	new_str = sympify(new_str,locals = locals)

	return new_str, sol 





"""-------- BOAT SIMULATION  ---------
_IN_
input: function that takes in time and states and returns a [3,1] vector for fx,fy,fz
time: 	[t_start, t_stop, t_step] -> start time, stop time, step length
init_cond: [x0, y0, yaw0, u0, v0, r0]

_OUT_
output: [u,v,r,du,dv,dr,fx,fy,fz,t], a [10,n] matrix

_NOTES_
Actual equation in component form:
du = 2.026e-4*fx + v*r - 0.0101*u -0.0492*np.abs(u)*u
dv = 2.026e-4*fy + u*r - 0.045*v - 0.4052*np.abs(v)*v
dr = 4.78e-5*fz -0.2595*(v+r) - 0.6027*np.abs(r)*r
"""
def boat_simulation(input, time = [0, 30, 0.01], init_cond = [0, 0, 0, 0, 0, 0]):

	## System ###
	def sys(t,X):

		
		tauB = np.zeros((3,1))
		tauB = input(t,X)

		# eta = np.zeros((3,1))
		# eta[0,0] = X[0]
		# eta[1,0] = X[1]
		# eta[2,0] = X[2]

		nu = np.zeros((3,1))
		nu[0,0] = X[3]
		nu[1,0] = X[4]
		nu[2,0] = X[5]

		#x = eta[0];
		#y = eta[1];
		yaw = X[2];


		u = nu[0];
		v = nu[1];
		r = nu[2];


		# mass and moment of inertia
		m = 4935.14
		Iz = 20928 #TODO Pz, sjekk at dette stemmer. Burde være Px?????

		# center of gravity
		xg = 0
		yg = 0

		# added mass
		Xdu = 0
		Ydv = 0
		Ydr = 0
		Ndv = 0
		Ndr = 0

		# damping (borrowed from Loe 2008)
		Xu = -50
		Yv = -200
		Yr = 0
		Nr = -1281

		Xuu = -135*1.8 #TODO gang med 1.8?? Hvorfor gjør Thomas det
		Yvv = -2000
		T = 4
		K = 0.5
		Nrr = -Iz*1.1374*K/T # TODO eller 0??

		# transformation matrix, equal to rotation matrix about z-axis
		J = np.array([[np.cos(yaw), -np.sin(yaw), 0],
		     [np.sin(yaw), np.cos(yaw),  0],
		     [0,        0,         1,]])

		# rigid body mass
		M_RB = np.array([[m, 0, 0],
		        [0, m, 0],
		        [0, 0, Iz]])

		# hydrodynamic added body mass
		M_A = -np.array([[Xdu, 0, 0],
		        [0, Ydv, Ydr],
		        [0, Ndv, Ndr]])
		 
		# total mass
		M = M_RB + M_A

		#Coriolis
		C_RB_g = np.zeros((3,3))
		C_RB_g[0,2] = -m*(xg*r+v)
		C_RB_g[2,0] = -C_RB_g[0,2]
		C_RB_g[1,2] = m*u
		C_RB_g[2,1] = -C_RB_g[1,2]

		C_A_g = np.zeros((3,3))
		C_A_g[0,2] = Ydv*v+Ydr*r
		C_A_g[2,0] = -C_A_g[0,2]
		C_A_g[1,2] = Xdu*u
		C_A_g[2,1] = -C_A_g[1,2]

		C_g = np.multiply(C_RB_g, C_A_g)

		#Linear damping
		Dl_g = -np.array([[Xu, 0, 0],
						[0, Yv, 0],
						[0, 0, Nr]])
		Dl_g[1,2] = -Yr;
		Dl_g[2,1] = -Nr;

		#Nonlinear damping
		Dnl_g = - np.array([[Xuu*np.abs(u), 0, 0],
						[0, Yvv*abs(v), 0],
						[Nrr*abs(r), 0, 0]])
		

		D_g = np.add(Dl_g, Dnl_g)

		

		eta_dot = np.dot(J,nu);
		nu_dot = np.dot(np.linalg.inv(M), (tauB - np.dot(C_g, nu) - np.dot(D_g, nu)))


		out = np.concatenate((eta_dot, nu_dot))
		

		return out

	#time
	t_start, t_stop, t_step = time 
	t = list(np.arange(t_start, t_stop, t_step))

	#solve
	sol = solve_ivp(sys, [t[0], t[-1]], init_cond, vectorized = True, max_step = t_step, t_eval = t)

	#input to the sys
	inp = np.zeros((3,len(sol.t)))
	for i,time in enumerate(sol.t):
		inp[:,i] = np.squeeze(input(time,sol.y[:,i]))

	#get the derivatives
	sol_dot = np.zeros((np.shape(sol.y)[0], np.shape(sol.y)[1]))
	for tmp, i in enumerate(sol.t):
		#sol_dot[:,tmp] = np.squeeze(sys(i, sol.y[:,tmp]))

		sol_dot[3,tmp] = 2.026e-4*inp[0,tmp] + sol.y[4,tmp]*sol.y[5,tmp] - 0.0101*sol.y[3,tmp] -0.0492*np.abs(sol.y[3,tmp])*sol.y[3,tmp]
		sol_dot[4,tmp] = 2.026e-4*inp[1,tmp] + sol.y[3,tmp]*sol.y[5,tmp] - 0.045*sol.y[4,tmp] - 0.4052*np.abs(sol.y[4,tmp])*sol.y[4,tmp]
		sol_dot[5,tmp]= 4.78e-5*inp[2,tmp] -0.2595*(sol.y[4,tmp]+sol.y[5,tmp]) - 0.6027*np.abs(sol.y[5,tmp])*sol.y[5,tmp]


	#final output [u,v,r,du,dv,dr,fx,fy,fz,t]
	#output = np.concatenate((sol_dot,inp,sol.t.reshape(1,-1)), axis = 0)
	output = np.concatenate((sol.y[3:],sol_dot[3:],inp,sol.t.reshape(1,-1)), axis = 0)


	#x-y plot
	plt.figure()
	plt.title('X-Y plot')
	plt.plot(sol.y[0], sol.y[1])


	return output

### takes in states as the output of boat_simulation and plots the states###
def boat_sim_plot(X, show = True):
	### plot ###
	# u v r
	plt.figure()
	plt.subplot(311)
	plt.plot(X[-1,:], X[0,:])
	plt.ylabel('u [m/s]')
	plt.grid()

	plt.subplot(312)
	plt.plot(X[-1,:], X[1,:])
	plt.ylabel('v [m/s]')
	plt.grid()

	plt.subplot(313)
	plt.plot(X[-1,:], X[2,:])
	plt.ylabel('r [m/s]')
	plt.xlabel('Time [s]')
	plt.grid()

	### plot ###
	# du dv dr
	plt.figure()
	plt.subplot(311)
	plt.plot(X[-1,:], X[3,:])
	plt.ylabel('du [m/s²]')
	plt.grid()

	plt.subplot(312)
	plt.plot(X[-1,:], X[4,:])
	plt.ylabel('dv [m/s²]')
	plt.grid()

	plt.subplot(313)
	plt.plot(X[-1,:], X[5,:])
	plt.ylabel('dr [m/s²]')
	plt.xlabel('Time [s]')
	plt.grid()

	#inputs
	plt.figure()
	plt.subplot(311)
	plt.plot(X[-1,:],X[6,:])
	plt.ylabel('tau x [N]')
	plt.grid()

	plt.subplot(312)
	plt.plot(X[-1,:],X[7,:])
	plt.ylabel('tau y [N]')
	plt.grid()

	plt.subplot(313)
	plt.plot(X[-1,:],X[8,:])
	plt.ylabel('tau z [Nm]')
	plt.xlabel('time [s]')
	plt.grid()


	if show:
		plt.show()

#plots the tree structure -useless
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




# """ ----------------- LEAST SQUARES ------------
# least squares that is much faster than the least squares regression previusly implemented
# Current implementation works with MSD system. calls on split_tree(), only works with arity 0 and 2 functions
# TODO:   - fix issues with singularity
# 		- make it compabile with arity 1 functions. 

# _IN_ 
# individual: 	the function 
# ddx,dx,x,tau: 	the data

# _OUT_
# mse: tuple with (mse,), the mean square error. 

# """

# def eval_fit(individual, ddx, dx, x, tau, pset):
# 	funcs, str_list = split_tree(individual, pset)
# 	F_list = []
# 	#print(individual)

# 	#top root is not 'add'
# 	if len(funcs) == 1:
# 		F = funcs[0](dx,x,tau)
# 		F_trans = np.transpose(F)
# 		p = np.dot(np.linalg.inv(np.dot(F_trans,F)),np.dot(F_trans,ddx))  # correct


# 	#top root is 'add'
# 	else:
# 		for func in funcs:
# 			F_list.append(func)
# 		F = np.zeros((len(ddx), len(F_list)))		
# 		for i, function in enumerate(F_list):
# 			F[:,i] = np.squeeze(function(dx,x,tau))

# 		#p = np.dot(np.dot(np.transpose(np.dot(np.linalg.pinv(F),F)),F),ddx)
# 		#p = np.dot(np.dot(np.linalg.pinv(F),F),np.dot(np.transpose(F),ddx)) #pseudo inverse? vet ikke hvordan det funker tbh
# 		#F_inv = np.linalg.pinv(F)
# 		F_trans = np.transpose(F)
# 		#p = np.dot(np.dot(np.dot(F_inv,F),F_trans),ddx)
# 		try:
# 			p = np.dot(np.linalg.inv(np.dot(F_trans,F)),np.dot(F_trans,ddx))  # correct
# 		except:
# 			print('Singular Matrix for individ:', individual)
# 			mse = 100 # large number
# 			return(mse,)

# 	tot_func = np.zeros((len(ddx), 1))
# 	for i, func in enumerate(funcs):
# 		tot_func = np.add(tot_func, p[i]*func(dx,x,tau))

# 	mse = math.fsum((ddx-tot_func)**2)/len(ddx)

# 	#show eq:
# 	if 0:
# 		locals = {
# 			'mul': lambda x, y : x * y,
# 			'add': lambda x, y : x + y,
# 			'add3': lambda x, y, z: x+y+z,
# 			'sub': lambda x, y : x - y,
# 			'protectedDiv': lambda x, y: x / y,
# 			'neg': lambda x: -x,
# 			'sin': lambda x: sin(x),
# 			'cos': lambda x: cos(x),
# 			'abs': lambda x: np.abs(x)#x if x >= 0 else -x
# 		}
# 		tot_str = ''
# 		for i, func_str in enumerate(str_list):
# 			tot_str = tot_str +'+'+ str(p[i][0])+ '*' +func_str
# 		print(sympify(tot_str,locals = locals))


# 		#plt.figure()
# 		#plt.title('mse: '+str(mse))
# 		#plt.plot(t,tot_func)
# 		#plt.plot(t,ddx)
# 		#plt.legend(['estimated', 'acctual'])
# 		#plt.show()
# 		#exit()

# 	return(mse,)


# def split_tree(individual, pset):
	
# 	def tree_trav(individual):
# 		nodes, edges, labels = gp.graph(individual)
# 		main_roots = []

# 		#is the first root add or sub
# 		if labels[0] == 'add':# or labels[0] == 'sub':
# 			main_roots.append(nodes[0])
# 		else:
# 			return None

# 		#find the main roots
# 		for node in sorted(nodes):
# 			if labels[node] == 'add':# or labels[node] == 'sub':
# 				if node not in main_roots:
# 					for edge in edges: 
# 						if node == edge[1] and edge[0] in main_roots: #if the previus node is in roots
# 							main_roots.append(node)

# 		for root in main_roots:
# 			for edge in edges:
# 				if edge[0] in main_roots:
# 					if edge[1] not in main_roots and edge[1] not in roots:					
# 						roots.append(edge[1])
# 		return main_roots

# 	def ext_funcs(individual):
# 		for root in roots:

# 			F = individual[individual.searchSubtree(root)]
# 			string = ' '
# 			for item in F:
# 				if item.arity == 2:
# 					if string[-1] == ')':
# 						string = string + ',' + item.name + '('	
# 					elif string[-1] == ' ':
# 						string = string + item.name + '('
# 					else:
# 						string = string + item.name +'('

# 				if item.arity == 0:
# 					if string[-1] == '(':
# 						string = string + item.format() +','
# 					elif string[-1] == ',':
# 						string = string + item.format() +')'
# 					elif string[-1] == ')':
# 						string = string +','+ item.format() +')'
# 					else: 
# 						string = string + item.format()

# 			#print('sub function: ',string)
# 			str_list.append(string)
# 			new_ind = gp.PrimitiveTree.from_string(string,pset)
# 			func1 = toolbox.compile(expr=new_ind)
# 			subtree_list.append(func1)
			
# 	subtree_list = []
# 	str_list = []
# 	roots = []
# 	main_roots = tree_trav(individual)
# 	if main_roots == None:
# 		str_list.append(str(individual))
# 		return [toolbox.compile(expr=individual)], str_list

# 	ext_funcs(individual)
# 	return subtree_list, str_list

# ######