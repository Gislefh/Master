import numpy as np
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from sklearn import preprocessing
import pydotplus
from PIL import Image
import io


##  -- import data -- 
file_path = '/home/gislehalv/Master/Data/data_fom_MatlabSim/du_data_wj3.csv'
data = np.loadtxt(file_path, delimiter = ',')
## data = [u,v,r,U,du,dv,dr,dU,force_x,force_y,force_z,time]


#X = np.concatenate((data[:, 0:3],data[:, 8:11]),axis = 1)  #[u,v,r,fx,fy,fz]

#Y = data[:, 4] #dv
#t = data[:, -1]

u = data[:, 0]
v = data[:, 1]
r = data[:, 2]
fx = data[:, 8]
fy = data[:, 9]
fz = data[:, 10]
du = data[:, 4]
dv = data[:, 5]
dr = data[:, 6]
t = data[:, -1]


du_eq = np.multiply(v,r) - 0.01 - 0.05 * np.multiply(np.absolute(u),u) + fx/5000
dv_eq = -4.24 * np.multiply(u,r) - 0.17 - 1.72 * np.multiply(np.absolute(u),v)
dr_eq = -0.26 * v - np.multiply(v,r) - 0.6 * np.multiply(np.absolute(r),r) + fz


plt.figure()
plt.subplot(311)
plt.plot(t,du)
plt.plot(t,du_eq)
plt.legend(['du', 'du_eq'])
plt.subplot(312)
plt.plot(t,dv)
plt.plot(t,dv_eq)
plt.legend(['dv', 'dv_eq'])
plt.subplot(313)
plt.plot(t,dr)
plt.plot(t,dr_eq)
plt.legend(['dr', 'dr_eq'])


#plt.show()

# nu = np.concatenate((u.reshape(-1,1),v.reshape(-1,1),r.reshape(-1,1)),axis = 1)
# tau = np.concatenate((fx.reshape(-1,1),fy.reshape(-1,1),fz.reshape(-1,1)),axis = 1)


# M = np.array([[4935, 0 ,0], [0, 4935, 0], [0, 0, 20928]])

# def C_func(nu, M):
# 	C = np.array([[0, 0, -M[1,1]*nu[1]-M[1,2]*nu[2]], [0, 0, M[0,0]*nu[0]], [M[1,1]*nu[1]+M[2,1]*nu[2], -M[0,0]*nu[0], 0]])
# 	return C

# def D_func(nu):
# 	D = np.array([[50+243*np.abs(nu[0]), 0, 0], [0, 200+2000*np.abs(nu[1]), 0], [0, 1281, 1281+2975*np.abs(nu[2])]])
# 	return D

# def C_func2(nu):
# 	m = 4925
# 	C = np.array([[0, 0, -m*nu[1]], [0, 0, m*nu[0]], [m*nu[1], -m*nu[0], 0]])
# 	return C

# d_nu = np.zeros((3, len(t)))
# M_inv = np.linalg.inv(M)

# for i in range(len(t)):
# 	#d_nu[:,i] = np.dot(M_inv,tau[i]) - np.dot(np.dot(M_inv, C_func2(nu[i])), nu[i]) - np.dot(np.dot(M_inv, D_func(nu[i])), nu[i])
# 	d_nu[:, i] = np.dot(M_inv, (tau[i] - np.dot(C_func2(nu[i]), nu[i]) - np.dot(D_func(nu[i]), nu[i]) ))

# plt.figure()
# plt.subplot(311)
# plt.plot(t,d_nu[0,:])
# plt.plot(t,du)
# plt.subplot(312)
# plt.plot(t,d_nu[1,:])
# plt.plot(t,dv)
# plt.subplot(313)
# plt.plot(t,d_nu[2,:])
# plt.plot(t,dr)





m = 4925

# du_eq2 = 1e-3 * 0.2026 * (fx + m*v*r - 50*u - 243*np.abs(u)*u)		# 2.026e-4*fx + v*r - 0.0101*u -0.0492*np.abs(u)*u # 
# dv_eq2 = 1e-3 * 0.2026 * (fy - m*u*r - 200*v - 2000*np.abs(v)*v)	# 2.026e-4*fy + u*r - 0.045*v - 0.4052*np.abs(v)*v # 
# dr_eq2 = 1e-3 * 0.0478 * (fz - 1281*v - 1281*r - 2975*np.abs(r)*r)	# wrong: 4.78e-5*fz -0.2595(v+r) - 0.6027*np.abs(r)*r     # 
																	# rigth: 4.78e-5*fz -0.0612(v+r) - 0.1422*np.abs(r)*r
du_eq2 = 2.026e-4*fx + v*r - 0.0101*u -0.0492*np.abs(u)*u
dv_eq2 = 2.026e-4*fy - u*r - 0.045*v - 0.4052*np.abs(v)*v 
dr_eq2 = 4.78e-5*fz -0.0612*(v+r) - 0.1422*np.abs(r)*r


plt.figure(66)
plt.subplot(311)
plt.plot(t,du_eq2)
plt.plot(t,du)
plt.legend(['du', 'du_eq'])

plt.subplot(312)
plt.plot(t,dv_eq2)
plt.plot(t,dv)
plt.legend(['dv', 'dv_eq'])

plt.subplot(313)
plt.plot(t,dr_eq2)
plt.plot(t,dr)
plt.legend(['dr', 'dr_eq'])



plt.figure(67)
plt.subplot(311)
plt.plot(t,fx)
plt.legend(['fx'])

plt.subplot(312)
plt.plot(t,fy)
plt.legend(['fy'])

plt.subplot(313)
plt.plot(t,fz)
plt.legend(['fz'])


plt.show()




exit()



###testing with scaling

du_scaled = du -np.mean(du)
du_scaled = du / np.std(du)


du_mean = np.mean(du)
du_std = np.std(du)

# plt.figure()
# plt.plot(t,du)
# plt.legend(['du_scaled'])

# plt.figure()
# plt.plot(t,du_eq2)
# plt.legend(['du_orig'])




X = np.concatenate((data[:, 0:3],data[:, 8:11]),axis = 1)  #[u,v,r,fx,fy,fz]

#zero mean unit variance
for i in range(np.shape(X)[1]):
	#if np.std(X[:, i]):
		#X[:, i] = X[:, i] - np.mean(X[:, i])
	X[:, i] = X[:, i] / np.std(X[:, i])

#
#X[:, 3] = X[:,3] /100

du_eq2_scaled = 2.026e-4*X[:,3] + X[:,1]*X[:,2] - 0.0101*X[:,0] -0.0492*np.abs(X[:,0])*X[:,0]

plt.figure()
plt.plot(t,du_scaled)
plt.legend('du_scaled')

plt.figure()
plt.plot(t,du_eq2)
plt.legend(['du_orig'])


plt.figure()
plt.plot(t,du_eq2_scaled)
plt.legend(['du_with_scaled_input'])

plt.show()




###------- does the shape of the eqation change or just the constants? 
plt.figure()
plt.subplot(312)
plt.plot(t,du_scaled)
plt.legend(['du_scaled'])


# a = 2.026e-4
# b = 0.0101
# c = 0.0492
best = 0

for a in np.logspace(-1,1,num = 50):
	print(a)
	for b in np.logspace(-2,-1,num = 50):
		for c in np.logspace(-1,1,num = 50):
			du_eq2_scaled = a*X[:,3] + X[:,1]*X[:,2] - b*X[:,0] - c*np.abs(X[:,0])*X[:,0]

			mse = ((du - du_eq2_scaled)**2).mean(axis=None)

			if not best:
				best = mse

			if mse < best:
				best = mse
				a_best,b_best,c_best = a,b,c

print('mse: ',best)
print('[a,b,c]: ',[a_best,b_best,c_best])


du_eq2_scaled = a_best*X[:,3] + X[:,1]*X[:,2] - b_best*X[:,0] - c_best*np.abs(X[:,0])*X[:,0]





plt.subplot(311)
plt.plot(t,du_scaled)
plt.legend(['du_scaled'])


plt.subplot(313)
plt.plot(t,du_eq2_scaled)
plt.legend(['du_with_scaled_input'])



plt.figure()
plt.plot(t,(du_eq2_scaled*du_std)+du_mean)
plt.plot(t,du)
plt.legend(['du with scaled input', 'du orig'])
plt.xlabel('Time [s]')
plt.ylabel('du [m/s²]')
plt.grid()


plt.show()

