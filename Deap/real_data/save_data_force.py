"""

make new datasets.  rpm and noz anlg to force

"""
import numpy as np
import matplotlib.pyplot as plt

load_path = '/home/gislehalv/Master/Data/numpy_data_from_bag/'
X1 = np.load(load_path + 'bag1_1.npy')
X2 = np.load(load_path + 'bag2_1.npy')
X3 = np.load(load_path + 'bag3_1.npy')
X4 = np.load(load_path + 'bag4_1.npy')




X = np.concatenate((X1,X2,X3,X4), axis = 1)
X[-1] = np.arange(0, len(X[0])*0.05, 0.05)

X[-3][X[-3] > 27] = 27 #remove error in the data
X[-3][X[-3] < -27] = -27

X[-3] = X[-3] * (np.pi/180)

# plt.figure()
# plt.plot(X[-1], X[-3])
# plt.show()
# exit()




### --- jet model ---
def jet_model(nu, jet_rpm, delta_nozzle):
	#constants
	lever_CGtowj_port = [-3.82, -0.475]
	lever_CGtowj_stb = [-3.82, 0.475]

	#rpm2thrust
	speed = nu[0] * 1.94384 # knots 
	a0 = 6244.15
	a1 = -178.46
	a2 = 0.881043
	thrust_unscaled = a0 + a1*speed + a2*(speed**2)

	r0 = 85.8316
	r1 = -1.7935
	r2 = 0.00533
	rpm_scale = 1/4530*(r0 + r1*jet_rpm + r2 * (jet_rpm **2))

	thrust = rpm_scale * thrust_unscaled * 0.5 ## 2 jets-fix


	#waterjet port
	Fx = thrust*np.cos(delta_nozzle)
	Fy = thrust*np.sin(delta_nozzle)

	#moment
	Nz_port = (lever_CGtowj_port[0]*Fy)- (lever_CGtowj_port[1]*Fx)
	Nz_stb = (lever_CGtowj_stb[0]*Fy)- (lever_CGtowj_stb[1]*Fx)

	tau_b =  [2*Fx, 2*Fy, Nz_port + Nz_stb]

	return tau_b

tauB = np.zeros((3, len(X[1])))

for i in range(np.shape(X)[1]):

	tauB[:, i] = jet_model(X[0:3, i], X[-4, i], X[-3, i])


# plt.figure()
# plt.subplot(311)
# plt.plot(X[-1], tauB[0, :])
# plt.ylabel('Fx [N]')
# plt.grid()

# plt.subplot(312)
# plt.plot(X[-1], tauB[1, :])
# plt.ylabel('Fy [N]')
# plt.grid()

# plt.subplot(313)
# plt.plot(X[-1], tauB[2, :])
# plt.ylabel('Nz [Nm]')
# plt.grid()
# plt.show()



X_new = np.concatenate((X[0:6], tauB, X[8:10]), axis = 0)



save_path = '/home/gislehalv/Master/Data/numpy_data_from_bag_force/'
save_as = 'all_bags_cut1.npy'
np.save(save_path+save_as, X_new)