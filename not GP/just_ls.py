"""
Testing out the implementation of eriksen and breivik
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import sympify, cos, sin, expand, collect, Lambda, lambdify, symbols
from scipy import optimize

if 0:
	X = np.load('/home/gislehalv/Master/Data/numpy_data_from_bag/' + 'bag1_025'+'.npy')


	U = np.sqrt(np.multiply(X[0],X[0]) + np.multiply(X[1],X[1])) # sqrt(u^2 + v^2)
	r = X[2]

	delta_t = np.divide(X[6],2000)
	delta_n = np.divide(X[7],27)

	delta_n[delta_n > 1] = 1 #remove error in the data
	delta_n[delta_n < -1] = -1

	#dU = #np.sqrt(np.multiply(X[3],X[3]) + np.multiply(X[4],X[4])) # sqrt(du^2 + dv^2)

	dU = np.diff(U) /0.05

	dU = np.append(dU,dU[-1])
	dr = X[5]

	data_time = X[-1]

	plt.figure()
	plt.subplot(311)
	plt.plot(data_time, r)
	plt.ylabel('r')
	plt.grid()
	plt.subplot(312)
	plt.plot(data_time, delta_n)
	plt.ylabel('delta_n')
	plt.grid()
	plt.subplot(313)
	plt.plot(data_time, delta_t)
	plt.ylabel('delta_t')
	plt.grid()
	plt.show()
	exit()


#### ---Make dataset----
def find_sigma():

	### ---- sigma U ---
	#steady state = [[from, to], [..]] 
	ss_b3 = np.array([[30,37], [45,57], [71,77], [145, 157], [180, 197]])
	ss_b1 = np.array([[660,667], [678, 682], [688, 697], [702, 712], [778, 787], [795, 803], [2422, 2495], [2045, 2083]])

	sigma_U_arr = []
	sigma_r_arr = []

	#load bag3
	X = np.load('/home/gislehalv/Master/Data/numpy_data_from_bag/' + 'bag3_025'+'.npy')
	U = np.sqrt(np.multiply(X[0],X[0]) + np.multiply(X[1],X[1])) # sqrt(u^2 + v^2)
	r = X[2]
	delta_t = np.divide(X[6],2000)
	delta_n = np.divide(X[7],27)
	delta_n[delta_n > 1] = 1 #remove error in the data
	delta_n[delta_n < -1] = -1
	dU = np.diff(U) /0.05
	dU = np.append(dU,dU[-1])
	dr = X[5]
	data_time = X[-1]

	for i in ss_b3:
		arg1 = np.argmin((data_time -i[0])**2)
		arg2 = np.argmin((data_time -i[1])**2)

		U_ss = U[arg1:arg2]
		r_ss = r[arg1:arg2]
		delta_t_ss = delta_t[arg1:arg2]
		delta_n_ss = delta_n[arg1:arg2]

		U_ss_mean = sum(U_ss)/len(U_ss)
		r_ss_mean = sum(r_ss)/len(r_ss)
		delta_t_ss_mean = sum(delta_t_ss)/len(delta_t_ss)
		delta_n_ss_mean = sum(delta_n_ss)/len(delta_n_ss)

		sigma_U_arr.append([U_ss_mean, r_ss_mean, delta_t_ss_mean, delta_n_ss_mean])


	#load bag1
	X = np.load('/home/gislehalv/Master/Data/numpy_data_from_bag/' + 'bag1_025'+'.npy')
	U = np.sqrt(np.multiply(X[0],X[0]) + np.multiply(X[1],X[1])) # sqrt(u^2 + v^2)
	r = X[2]
	delta_t = np.divide(X[6],2000)
	delta_n = np.divide(X[7],27)
	delta_n[delta_n > 1] = 1 #remove error in the data
	delta_n[delta_n < -1] = -1
	dU = np.diff(U) /0.05
	dU = np.append(dU,dU[-1])
	dr = X[5]
	data_time = X[-1]

	for i in ss_b1:
		arg1 = np.argmin((data_time -i[0])**2)
		arg2 = np.argmin((data_time -i[1])**2)

		U_ss = U[arg1:arg2]
		r_ss = r[arg1:arg2]
		delta_t_ss = delta_t[arg1:arg2]
		delta_n_ss = delta_n[arg1:arg2]

		U_ss_mean = sum(U_ss)/len(U_ss)
		r_ss_mean = sum(r_ss)/len(r_ss)
		delta_t_ss_mean = sum(delta_t_ss)/len(delta_t_ss)
		delta_n_ss_mean = sum(delta_n_ss)/len(delta_n_ss)

		sigma_r_arr.append([U_ss_mean, r_ss_mean, delta_t_ss_mean, delta_n_ss_mean])
	



	### --- sigma r --

	ss_b1 = np.array([[178, 183], [209, 215], [238, 242], [267, 277], [303, 307], [331, 337], [1654, 1660], [1670, 1675], [1685, 1691], [1700, 1707], [1717, 1722], [1974, 1979], [1986, 1993], [2000, 2004], [2010, 2017], [2023, 2029], [2334, 2496], [2509, 2526], [2530, 2542], [2551, 2585], [2591, 2604], [2613, 2646], [2653, 2667]])

	for i in ss_b1:
		arg1 = np.argmin((data_time -i[0])**2)
		arg2 = np.argmin((data_time -i[1])**2)

		U_ss = U[arg1:arg2]
		r_ss = r[arg1:arg2]
		delta_t_ss = delta_t[arg1:arg2]
		delta_n_ss = delta_n[arg1:arg2]

		U_ss_mean = sum(U_ss)/len(U_ss)
		r_ss_mean = sum(r_ss)/len(r_ss)
		delta_t_ss_mean = sum(delta_t_ss)/len(delta_t_ss)
		delta_n_ss_mean = sum(delta_n_ss)/len(delta_n_ss)

		sigma_U_arr.append([U_ss_mean, r_ss_mean, delta_t_ss_mean, delta_n_ss_mean])
	



	return sigma_U_arr, sigma_r_arr

sigma_U_data, sigma_r_data = find_sigma()

def find_mu_mr():

	### -------- bag4 ----------
	X = np.load('/home/gislehalv/Master/Data/numpy_data_from_bag/' + 'bag4_025'+'.npy')
	U = np.sqrt(np.multiply(X[0],X[0]) + np.multiply(X[1],X[1])) # sqrt(u^2 + v^2)
	r = X[2]
	delta_t = np.divide(X[6],2000)
	delta_n = np.divide(X[7],27)
	delta_n[delta_n > 1] = 1 #remove error in the data
	delta_n[delta_n < -1] = -1
	dU = np.diff(U) /0.05
	dU = np.append(dU,dU[-1])
	dr = X[5]
	data_time = X[-1]

	###finding m_U, steps = [[start, stop, start_period_end, stop_period_start],[]..
	steps_U = [[20, 55, 27, 45], [50, 75,  57, 70], [80, 115, 87, 112], [112, 145, 117, 144], [145, 175, 147, 174], [174, 200, 177, 193], [200, 235, 207, 220], [220, 280, 237, 265], [280, 327, 297, 325], [325, 341, 327, 338]] ## bag4

	data_from_steps_U = []
	for step_time_arr in steps_U:
		U_step = []
		delta_t_step = []
		r_step = []
		for i, t in enumerate(data_time):
			if t > step_time_arr[0] and t < step_time_arr[1]: # a step
				U_step.append([U[i], t])
				delta_t_step.append([delta_t[i], t])
				r_step.append([r[i], t])
			


		s_UiP = []
		s_UiN = []
		UiN = []
		UiP = []
		tau_miN = []
		riP = []
		riN = []

		for i in range(len(U_step)):
			if U_step[i][1] < step_time_arr[2]:
				s_UiN.append(delta_t_step[i][0])
				UiN.append(U_step[i][0])
				riN.append(r_step[i][0])
				tau_miN.append(delta_t_step[i][0])
			if U_step[i][1] > step_time_arr[3]:
				UiP.append(U_step[i][0])
				s_UiP.append(delta_t_step[i][0])
				riP.append(r_step[i][0])


		s_UiP = sum(s_UiP)/len(s_UiP)
		s_UiN = sum(s_UiN)/len(s_UiN)

		riP = sum(riP)/len(riP)
		riN = sum(riN)/len(riN)

		UiP = sum(UiP)/len(UiP)
		UiN = sum(UiN)/len(UiN)

	 
		ki = (s_UiP - s_UiN) / (UiP - UiN)




		U_step = np.array(U_step)
		delta_Ui = np.subtract(U_step[:,0], UiN)
		

		d_delta_Ui = np.diff(delta_Ui)/(data_time[1]-data_time[0])
		d_delta_Ui = np.append(d_delta_Ui, d_delta_Ui[-1]) 
		
		tau_miN = sum(tau_miN)/len(tau_miN)
		delta_t_step = np.array(delta_t_step)
		delta_tau_mi = np.subtract(delta_t_step[:,0], tau_miN)

		def sys(mui):
			LHS =  mui* d_delta_Ui + ki*delta_Ui 
			RHS = delta_tau_mi
			se = np.square(LHS-RHS)
			#print(sum(se)/len(se))
			return se


		x0 = 1
		sol = optimize.least_squares(sys,x0)
		#print(sol.x, sol.cost)


		#x_out = [(UiP+UiN)/2, (riP+riN)/2]
		data_from_steps_U.append([(UiP+UiN)/2, (riP+riN)/2, sol.x[0]]) 

	print()
	## ---- bag1 ------
	X = np.load('/home/gislehalv/Master/Data/numpy_data_from_bag/' + 'bag1_025'+'.npy')
	U = np.sqrt(np.multiply(X[0],X[0]) + np.multiply(X[1],X[1])) # sqrt(u^2 + v^2)
	r = X[2]
	delta_t = np.divide(X[6],2000)
	delta_n = np.divide(X[7],27)
	delta_n[delta_n > 1] = 1 #remove error in the data
	delta_n[delta_n < -1] = -1
	dU = np.diff(U) /0.05
	dU = np.append(dU,dU[-1])
	dr = X[5]
	data_time = X[-1]

	###finding m_r, steps = [[start, stop, start_period_end, stop_period_start],[]..
	steps_r = [[820, 835, 827, 834], [834, 850, 834.5, 846], [847, 870, 850, 865], [870, 885, 871, 882]]

	data_from_steps_r = []
	for step_time_arr in steps_r:
		U_step = []
		delta_n_step = []
		r_step = []
		for i, t in enumerate(data_time):
			if t > step_time_arr[0] and t < step_time_arr[1]: # a step
				U_step.append([U[i], t])
				delta_n_step.append([delta_n[i], t])
				r_step.append([r[i], t])
			


		s_riP = []
		s_riN = []
		UiN = []
		UiP = []
		tau_miN = []
		riP = []
		riN = []

		for i in range(len(U_step)):
			if r_step[i][1] < step_time_arr[2]:
				s_riN.append(delta_n_step[i][0])
				UiN.append(U_step[i][0])
				riN.append(r_step[i][0])
				tau_miN.append(delta_n_step[i][0])
			if r_step[i][1] > step_time_arr[3]:
				UiP.append(U_step[i][0])
				s_riP.append(delta_n_step[i][0])
				riP.append(r_step[i][0])

		s_riP = sum(s_riP)/len(s_riP)
		s_riN = sum(s_riN)/len(s_riN)

		riP = sum(riP)/len(riP)
		riN = sum(riN)/len(riN)

		UiP = sum(UiP)/len(UiP)
		UiN = sum(UiN)/len(UiN)

	 
		ki = (s_riP - s_riN) / (riP - riN)

		r_step = np.array(r_step)
		delta_ri = np.subtract(r_step[:,0], riN)

		d_delta_ri = np.diff(delta_ri)/(data_time[1]-data_time[0])
		d_delta_ri = np.append(d_delta_ri, d_delta_ri[-1]) 


		tau_miN = sum(tau_miN)/len(tau_miN)
		delta_n_step = np.array(delta_n_step)
		delta_tau_ri = np.subtract(delta_n_step[:,0], tau_miN)


		def sys(mui):
			LHS =  mui* d_delta_ri + ki*delta_ri 
			RHS = delta_tau_ri
			se = np.square(LHS-RHS)
			return se

		x0 = 1
		sol = optimize.least_squares(sys,x0)
		#print(sol.x, sol.cost)

		#x_out = [, ]
		data_from_steps_r.append([(UiP+UiN)/2, (riP+riN)/2, sol.x[0]]) 



	return data_from_steps_U, data_from_steps_r
m_U_data, m_r_data = find_mu_mr()



###---- prepro

##### Least squares
def ls():
	func_sigma = ['U', 'r', 'U**2', 'U*r', 'r**2', 'U**3', 'U**2*r', 'U*r**2', 'r**3', 'U**4', 'U**3*r', 'U**2*r**2', 'U*r**3', 'r**4'] 
	func_m = ['U', 'r', 'U**2', 'U*r', 'r**2', 'U**3', 'U**2*r', 'U*r**2', 'r**3', 'U**4', 'U**3*r', 'U**2*r**2', 'U*r**3', 'r**4'] ## tanh(a(U-b))

	def str2func(function):
		U, r  = symbols('U r')
		f = lambdify((U,r),function, 'numpy')
		return f

	sigma_U = []
	sigma_r = []
	for i in func_sigma:
		sigma_r.append(str2func(i))
		sigma_U.append(str2func(i))
	m_U = []
	m_r = []
	for i in func_m:
		m_U.append(str2func(i))
		m_r.append(str2func(i))


	#data
	X1 = np.zeros((2, len(sigma_U_data)))
	Y1 = np.zeros((len(sigma_U_data), 1))

	X2 = np.zeros((2,  len(sigma_r_data)))
	Y2 = np.zeros((len(sigma_r_data), 1))

	X3 = np.zeros((2, len(m_U_data)))
	Y3 = np.zeros((len(m_U_data),1))

	X4 = np.zeros((2, len(m_r_data)))
	Y4 = np.zeros((len(m_r_data), 1))

	for i, data in enumerate(sigma_U_data):
		X1[:, i] = data[0:2] 
		Y1[i, :] = data[2]

	for i, data in enumerate(sigma_r_data):
		X2[:, i] = data[0:2] 
		Y2[i, :] = data[3]

	for i, data in enumerate(m_U_data):
		X3[:, i] = data[0:2] 
		Y3[i, :] = data[2]
		
	for i, data in enumerate(m_r_data):
		X4[:, i] = data[0:2] 
		Y4[i, :] = data[2]

	F1 = np.zeros((np.shape(X1)[1], len(sigma_U)))
	for i in range(np.shape(F1)[0]):
		for j, fun in enumerate(sigma_U):
			F1[i,j] = fun(X1[0,i], X1[1,i])

	F2 = np.zeros((np.shape(X2)[1], len(sigma_r) ))
	for i in range(np.shape(F2)[0]):
		for j, fun in enumerate(sigma_r):
			F2[i,j] = fun(X2[0,i], X2[1,i])

	F3 = np.zeros((np.shape(X3)[1], len(m_U) ))
	for i in range(np.shape(F3)[0]):
		for j, fun in enumerate(sigma_U):
			F3[i,j] = fun(X3[0,i], X3[1,i])

	F4 = np.zeros((np.shape(X4)[1], len(m_r) ))
	for i in range(np.shape(F4)[0]):
		for j, fun in enumerate(m_r):
			F4[i,j] = fun(X4[0,i], X4[1,i])

	F1_trans = np.transpose(F1)
	beta1 = np.dot(np.linalg.inv(np.dot(F1_trans,  F1)), np.dot(F1_trans, Y1))

	F2_trans = np.transpose(F2)
	beta2 = np.dot(np.linalg.inv(np.dot(F2_trans,  F2)), np.dot(F2_trans, Y2))

	F3_trans = np.transpose(F3)
	beta3 = np.dot(np.linalg.inv(np.dot(F3_trans,  F3)), np.dot(F3_trans, Y3))

	F4_trans = np.transpose(F4)
	beta4 = np.dot(np.linalg.inv(np.dot(F4_trans,  F4)), np.dot(F4_trans, Y4))

	beta = np.concatenate((beta1, beta2, beta3, beta4), axis = 1)

	
	func_list = [sigma_U, sigma_r, m_U, m_r]

	return beta, func_list




beta, func_list = ls()


def sim_result(beta, func_list):
	#load bag
	X = np.load('/home/gislehalv/Master/Data/numpy_data_from_bag/' + 'bag1_025'+'.npy')
	U = np.sqrt(np.multiply(X[0],X[0]) + np.multiply(X[1],X[1])) # sqrt(u^2 + v^2)
	r = X[2]
	delta_t = np.divide(X[6],2000)
	delta_n = np.divide(X[7],27)
	delta_n[delta_n > 1] = 1 #remove error in the data
	delta_n[delta_n < -1] = -1
	dU = np.diff(U) /0.05
	dU = np.append(dU,dU[-1])
	dr = X[5]
	data_time = X[-1]

	#s_U = np.zeros((len(data_time)))
	#s_r = np.zeros((len(data_time)))
	#m_U = np.zeros((len(data_time)))
	#m_r = np.zeros((len(data_time)))	




	sys = np.zeros((2,len(U)))
	for i in range(len(U)):
		m_U = 0
		m_r = 0
		s_U = 0
		s_r = 0
		for j, func in enumerate(func_list[0]):
			s_U = s_U + beta[j,0] *func(U[i], r[i])

		for j, func in enumerate(func_list[1]):
			s_r = s_r + beta[j,1] *func(U[i], r[i])

		for j, func in enumerate(func_list[2]):
			m_U = m_U + beta[j,2] *func(U[i], r[i])

		for j, func in enumerate(func_list[3]):
			m_r = m_r + beta[j,3] *func(U[i], r[i])


		M = np.array([[m_U, 0],[0, m_r]])
		S = np.array([[s_U],[s_r]])
		tau = np.array([[delta_t[i]],[delta_n[i]]])
		sys[:,i] = np.squeeze(np.dot(np.linalg.inv(M),np.subtract(tau, S)))

	plt.figure()
	plt.plot(data_time ,sys[0,:])
	plt.plot(data_time, dU)
	plt.legend(['pred', 'dU'])
	plt.figure()
	plt.plot(data_time, sys[1,:])
	plt.plot(data_time, dr)
	plt.legend(['pred', 'dr'])
	plt.show()
sim_result(beta, func_list)


def opt_ls():
	func_sigma = ['1','U', 'r', 'U**2', 'U*r', 'r**2', 'U**3', 'U**2*r', 'U*r**2', 'r**3', 'U**4', 'U**3*r', 'U**2*r**2', 'U*r**3', 'r**4'] 
	func_m = ['1','U', 'r', 'U**2', 'U*r', 'r**2', 'U**3', 'U**2*r', 'U*r**2', 'r**3', 'U**4', 'U**3*r', 'U**2*r**2', 'U*r**3', 'r**4'] ## tanh(a(U-b))

	def str2func(function):
		U, r = symbols('U r ')
		f = lambdify((U,r),function, 'numpy')
		return f


	m_U = []
	for part in func_sigma:
		m_U.append(str2func(part))
	m_r = []
	for part in func_sigma:
		m_r.append(str2func(part))
	sigma_U = []
	for part in func_m:
		sigma_U.append(str2func(part))
	sigma_r = []
	for part in func_m:
		sigma_r.append(str2func(part))

	y = np.array([dU,dr])

	##assuming func_sig = func_m
	def f_1(beta):
		mU, mr, sigmaU, sigmar = 0,0,0,0
		for i, func in enumerate(m_U):
			mU = mU +func(U,r)*beta[i]
			mr = mr +func(U,r)*beta[i+len(m_U)]
			sigmaU = sigmaU + func(U,r)*beta[i+2*len(m_U)]
			sigmar = sigmar + func(U,r)*beta[i+3*len(m_U)] 

		sys = np.zeros((2,len(U)))
		for i in range(len(U)):

			M = np.array([[mU[i], 0],[0, mr[i]]])
			S = np.array([[sigmaU[i]],[sigmar[i]]])
			tau = np.array([[delta_t[i]],[delta_n[i]]])
			sys[:,i] = np.squeeze(np.dot(np.linalg.inv(M),np.subtract(tau, S)))

		se = np.square(y-sys)
		print('mse: ',math.fsum(se[0,:])/len(U))
		return se[0,:]
	def f_2(beta):
		mU, mr, sigmaU, sigmar = 0,0,0,0
		for i, func in enumerate(m_U):
			mU = mU +func(U,r)*beta[i]
			mr = mr +func(U,r)*beta[i+len(m_U)]
			sigmaU = sigmaU + func(U,r)*beta[i+2*len(m_U)]
			sigmar = sigmar + func(U,r)*beta[i+3*len(m_U)] 

		sys = np.zeros((2,len(U)))
		for i in range(len(U)):

			M = np.array([[mU[i], 0],[0, mr[i]]])
			S = np.array([[sigmaU[i]],[sigmar[i]]])
			tau = np.array([[delta_t[i]],[delta_n[i]]])
			sys[:,i] = np.squeeze(np.dot(np.linalg.inv(M),np.subtract(tau, S)))


		se = np.square(y-sys)
		print('mse: ',math.fsum(se[0,:])/len(U))
		return se[1,:]


	x0 = np.zeros((len(m_U)+len(m_r)+len(sigma_U)+len(sigma_r))) +1
	sol_1 = optimize.least_squares(f_1,x0, max_nfev = 300)
	print('sol_1:',  sol_1.x)
	sol_2 = optimize.least_squares(f_2,x0, max_nfev = 300)
	print('sol_2:',  sol_2.x)

	def plot_sol(sol1, sol2):
		mU, mr, sigmaU, sigmar = 0,0,0,0
		for i, func in enumerate(m_U):
			mU = mU +func(U,r)*sol1[i]
			mr = mr +func(U,r)*sol2[i+2*len(m_U)]
			sigmaU = sigmaU + func(U,r)*sol1[i+len(m_U)]
			sigmar = sigmar + func(U,r)*sol2[i+3*len(m_U)] 

		sys = np.zeros((2,len(U)))
		for i in range(len(U)):

			M = np.array([[mU[i], 0],[0, mr[i]]])
			S = np.array([[sigmaU[i]],[sigmar[i]]])
			tau = np.array([[delta_t[i]],[delta_n[i]]])
			sys[:,i] = np.squeeze(np.dot(np.linalg.inv(M),np.subtract(tau, S)))


		plt.figure()
		plt.plot(data_time, sys[0,:])
		plt.plot(data_time, y[0,:])
		plt.legend(['pred U', 'sol'])
		plt.grid()

		plt.figure()
		plt.plot(data_time, sys[1,:])
		plt.plot(data_time, y[1,:])
		plt.legend(['pred r', 'sol'])
		plt.grid()

		plt.show()
	plot_sol(sol_1.x, sol_2.x)


#opt_ls()


"""
	#tot_fun = np.concatenate((sigma_U, sigma_r, m_U, m_r))

	# X = np.zeros((2, len(sigma_U_data) + len(sigma_r_data) + len(m_U_data) + len(m_r_data) ))
	# Y = np.zeros((1, len(sigma_U_data) + len(sigma_r_data) + len(m_U_data) + len(m_r_data) ))
	# for i, data in enumerate(sigma_U_data):
	# 	X[:, i] = data[0:2] 
	# 	Y[:, i] = data[2]

	# for i, data in enumerate(sigma_r_data):
	# 	X[:, i+len(sigma_U_data)] = data[0:2] 
	# 	Y[:, i+len(sigma_U_data)] = data[2]

	# for i, data in enumerate(m_U_data):
	# 	X[:, i+len(sigma_U_data) + len(sigma_r_data)] = data[0:2] 
	# 	Y[:, i+len(sigma_U_data) + len(sigma_r_data)] = data[2]
		
	# for i, data in enumerate(m_r_data):
	# 	X[:, i+len(sigma_U_data) + len(sigma_r_data) + len(m_U_data)] = data[0:2] 
	# 	Y[:, i+len(sigma_U_data) + len(sigma_r_data) + len(m_U_data)] = data[2]

	# F = np.zeros((np.shape(X)[1], len(tot_fun) ))
	# for i in range(np.shape(X)[1]):
	# 	for j, fun in enumerate(tot_fun):
	# 		F[i,j] = fun(X[0,i], X[1,i])

	# F_trans = np.transpose(F)


	# print(np.linalg.matrix_rank(np.diagflat([1]*100)))
	# exit()
	# for i in range(np.shape(F)[0]):
	# 	F = F[0:i]	
	# 	F_trans = np.transpose(F)
	# 	det = np.linalg.det(np.dot(F_trans, F))
	# 	if det != 0:
	# 		print(det)

	# beta = np.dot(np.linalg.inv(np.dot(F_trans,  F), np.dot(F_trans, Y)))

	# print(beta)

"""