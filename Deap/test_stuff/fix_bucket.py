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


import rosbag
import time
from scipy import interpolate
from pytictoc import TicToc

bag_1 = 'hal_control_2018-12-11-10-53-26_0' #large!
bag_2 = 'hal_control_2018-12-11-11-49-22_0' #similar to bag1 but smaller
bag_3 = 'hal_control_2018-12-11-12-13-58_0' # speed steps
bag_4 = 'hal_control_2018-12-11-12-13-58_0' # speed steps



# bag path
path = '/home/gislehalv/Master/Data/'


bagFile_path_train = path + bag_2 + '.bag'
bagFile_path_test = path + bag_1 + '.bag'


def open_bag(path, plot = False, thr_bucket = True):
	tic_t = TicToc()
	tic_t.tic()
	bag = rosbag.Bag(path)
	bagContents = bag.read_messages()
	if not bagContents:
		print('bag is empty')
		exit()
	print('open bag time')
	tic_t.toc()

	#save?
	save = False

	#listOfTopics = []
	#for topic, msg, t in bagContents:
	#	if topic not in listOfTopics:
	#		listOfTopics.append(topic)
	#print('TOPICS:')
	#print(listOfTopics)


	#### -- jet data --
	tic_t.tic()
	cnt_jet = 0
	for subtopic, msg, t in bag.read_messages('/usv/hamjet/status_high'):
		cnt_jet += 1

	jet_data  = np.zeros((4,cnt_jet))
	i = 0
	for subtopic, msg, t in bag.read_messages('/usv/hamjet/status_high'):
		# engines rpm, steering, jet time, bucket
		jet_data[0, i] = (msg.port_shaft_rpm + msg.stbd_shaft_rpm) / 2
		jet_data[1, i] = (msg.port_steering + msg.stbd_steering) / 2
		jet_data[2, i] = t.to_sec()
		jet_data[3, i] = (msg.port_reverse + msg.stbd_reverse) / 2
		i += 1

	# Set initail jet time to zero
	jet_data[2, :] = jet_data[2, :] - jet_data[2, 0]
	print('open jet data time')
	tic_t.toc()
	#### -- nav data --
	tic_t.tic()
	cnt_nav = 0
	for subtopic, msg, t in bag.read_messages('/usv/navp_msg'):
		cnt_nav += 1

	nav_data = np.zeros((7,cnt_nav))
	i = 0
	for subtopic, msg, t in bag.read_messages('/usv/navp_msg'):
		#surge, sway, yaw
		nav_data[0, i] = msg.pose.latitude
		nav_data[1, i] = msg.pose.longitude
		nav_data[2, i] = msg.pose.heading

		#surge, sway, yaw - Rate
		nav_data[3, i] = msg.vel.xVelocityB
		nav_data[4, i] = msg.vel.yVelocityB
		nav_data[5, i] = msg.rate.zAngularRateB #prob in deg/s

		# nav time
		nav_data[6, i] = t.to_sec()
		i += 1

	#set initial time to zero
	nav_data[6, :] = nav_data[6, :] - nav_data[6, 0] 
	print('open nav data time')
	tic_t.toc()
	tic_t.tic()
	####		 --Fiter--
	#set the sample time to 0.05
	def filter(nav_data):
		#savgol_filter
		sav_fil = False
		FB_fil = False
		spline = False
		interpol = True

		#savgol_filter
		if sav_fil:
			# u_smooth_deriv = signal.savgol_filter(nav_data[3, :], 51, 4, deriv=1, delta=nav_data[3, 1]- nav_data[3, 0]) 
			# v_smooth_deriv = signal.savgol_filter(nav_data[4, :], 51, 4, deriv=1, delta=nav_data[3, 1]- nav_data[3, 0]) 
			# r_smooth_deriv = signal.savgol_filter(nav_data[5, :], 51, 4, deriv=1, delta=nav_data[3, 1]- nav_data[3, 0]) 

			u_smooth = signal.savgol_filter(nav_data[3, :], 51, 4) 
			v_smooth = signal.savgol_filter(nav_data[4, :], 51, 4) 
			r_smooth = signal.savgol_filter(nav_data[5, :], 51, 4) 

		#find freq
		if 0:
			len_ = 30
			test_t = list(np.arange(0, len_, 0.01))
			sp = np.fft.rfft(np.sin(test_t))
			freq = np.fft.rfftfreq(len(test_t))
			plt.figure()
			plt.plot(freq, sp.real)
			plt.grid()
			plt.ylabel('amount?')
			plt.xlabel('freq')

			#butter
			order = 2
			cuttoff = 0.003
			b, a = signal.butter(order, cuttoff, btype='low', analog=False, output='ba')

			#Forward -Backward filter
			if FB_fil:
				sin = signal.filtfilt(b, a, np.sin(test_t)) 
		 

			plt.figure()
			plt.plot(test_t, sin)
			plt.plot(test_t, np.sin(test_t))
			plt.legend(['butter', 'orig'])

			plt.show()
			exit()
		
		if FB_fil:
			#butter
			order = 2
			cuttoff = 0.05
			b, a = signal.butter(order, cuttoff, btype='low', analog=False, output='ba')

			#Forward -Backward filter
			u_smooth = signal.filtfilt(b, a, nav_data[3, :]) 
			v_smooth = signal.filtfilt(b, a, nav_data[4, :]) 
			r_smooth = signal.filtfilt(b, a, nav_data[5, :]) 

			#forward
			# u_smooth = signal.lfilter(b, a, nav_data[3, :]) 
			# v_smooth = signal.lfilter(b, a, nav_data[4, :]) 
			# r_smooth = signal.lfilter(b, a, nav_data[5, :]) 
		
		#spline 
		if spline: #NOPE
			u_smooth = signal.gauss_spline(nav_data[3, :], 5) 
			v_smooth = signal.gauss_spline(nav_data[4, :], 5) 
			r_smooth = signal.gauss_spline(nav_data[5, :], 5) 

		#interpolate then smooth
		if interpol:
			steps = 0.05
			interp_arr = list(np.arange(nav_data[6, 0], nav_data[6, -1], steps))
			u_int = np.interp(interp_arr, nav_data[6, :],nav_data[3, :])
			v_int = np.interp(interp_arr, nav_data[6, :],nav_data[4, :])
			r_int = np.interp(interp_arr, nav_data[6, :],nav_data[5, :])

			order = 2
			cuttoff = 0.3
			b, a = signal.butter(order, cuttoff, btype='low', analog=False, output='ba')

			#Forward -Backward filter
			u_smooth = signal.filtfilt(b, a, u_int) 
			v_smooth = signal.filtfilt(b, a, v_int) 
			r_smooth = signal.filtfilt(b, a, r_int) 
			return u_smooth, v_smooth, r_smooth, interp_arr
			#plot
			if 0:
				plt.figure()
				plt.plot(nav_data[6, :], nav_data[3, :])
				plt.plot(interp_arr, u_int)
				plt.plot(interp_arr, u_smooth)
				plt.legend([ 'orig', 'int', 'int smooth'])
				plt.ylabel('u')
				plt.grid()

				plt.figure()
				plt.plot(nav_data[6, :], nav_data[4, :])
				plt.plot(interp_arr, v_int)
				plt.plot(interp_arr, v_smooth)
				plt.legend([ 'orig', 'int', 'int smooth'])
				plt.ylabel('v')
				plt.grid()

				plt.figure()
				plt.plot(nav_data[6, :], nav_data[5, :])
				plt.plot(interp_arr, r_int)
				plt.plot(interp_arr, r_smooth)
				plt.legend([ 'orig', 'int', 'int smooth'])
				plt.ylabel('r')
				plt.grid()

				plt.show()
		
		return u_smooth, v_smooth, r_smooth
	u_smooth, v_smooth, r_smooth, interp_arr = filter(nav_data)


	#### 		-- Integrate --
	def integrate(nav_data, u_smooth, v_smooth, r_smooth):
		du = np.diff(nav_data[3, :],n = 1)
		dv = np.diff(nav_data[4, :],n = 1)
		dr = np.diff(nav_data[5, :],n = 1)

		du_smo = np.diff(u_smooth ,n = 1)
		dv_smo = np.diff(v_smooth ,n = 1)
		dr_smo = np.diff(r_smooth ,n = 1)

		#add the last signal twice 
		du = np.concatenate((du,[du[-1]]))
		dv = np.concatenate((dv,[dv[-1]]))
		dr = np.concatenate((dr,[dr[-1]]))

		du_smo = np.concatenate((du_smo,[du_smo[-1]]))
		dv_smo = np.concatenate((dv_smo,[dv_smo[-1]]))
		dr_smo = np.concatenate((dr_smo,[dr_smo[-1]]))

		return du, dv, dr, du_smo, dv_smo, dr_smo
	du, dv, dr, du_smo, dv_smo, dr_smo =  integrate(nav_data, u_smooth, v_smooth, r_smooth)


	def interpolate(jet_data, interp_arr):
		jet_rpm = 		np.interp(interp_arr, jet_data[2, :],jet_data[0, :])
		nozzle_angle =	np.interp(interp_arr, jet_data[2, :],jet_data[1, :]) # not really nozzle angle but rather [-100, 100]% = [-27, 27] deg 
		bucket = 		np.interp(interp_arr, jet_data[2, :],jet_data[3, :]) # from [-100, to 100]
		return jet_rpm, nozzle_angle, bucket
	jet_rpm, nozzle_angle, bucket = interpolate(jet_data, interp_arr)

	#preeprossesed matrix of variables.
	X = [u_smooth, v_smooth, r_smooth, du_smo, dv_smo, dr_smo, jet_rpm, nozzle_angle, bucket, interp_arr]
	X = np.array(X)

	# a test to check if the bucket is fully open in all the data. if not - start where it becomes > 95 and end if it <95
	def bucket_fully_open(X):
		if X[-2, 0] > 95:
			start = 0
		else:
			start = -1
		for i in range(np.shape(X)[1]):
			if X[-2, i] > 95 and start == -1:
				start = i
			if X[-2, i] < 95 and start == -1:
				continue 
			if X[-2, i] < 95 and start != -1:
				stop = i
				break

		X = X[:,start:stop]
		return X
	
	if thr_bucket: #use the  bucket_fully_open function 
		X = bucket_fully_open(X)
	print('rest time')
	tic_t.toc()
	### ---Plots----
	if plot:
		plt.figure()
		plt.subplot(311)
		plt.plot(jet_data[2, :], jet_data[0, :])
		plt.ylabel('RPM')
		plt.subplot(312)
		plt.plot(jet_data[2, :], jet_data[1, :])
		plt.ylabel('Nozzle angle')
		plt.subplot(313)
		plt.plot(jet_data[2, :], jet_data[3, :])
		plt.ylabel('Bucket')
		plt.grid()

		plt.figure()
		plt.subplot(311)
		plt.plot(nav_data[6, :], du)
		plt.plot(interp_arr, du_smo)
		plt.legend(['du', 'du_smooth'])
		plt.grid()
		plt.subplot(312)
		plt.plot(nav_data[6, :],dv)
		plt.plot(interp_arr,dv_smo)
		plt.legend(['dv', 'dv_smooth'])
		plt.grid()
		plt.subplot(313)
		plt.plot(nav_data[6, :],dr)
		plt.plot(interp_arr,dr_smo)
		plt.legend(['dr', 'dr_smooth'])
		plt.grid()

		plt.figure()
		plt.subplot(311)
		plt.plot(nav_data[6, :], nav_data[3, :], 'r.-')
		plt.plot(interp_arr, u_smooth)
		plt.ylabel('u')
		plt.grid()
		plt.subplot(312)
		plt.plot(nav_data[6, :], nav_data[4, :], 'r.-')
		plt.plot(interp_arr, v_smooth)
		plt.ylabel('v')
		plt.grid()
		plt.subplot(313)
		plt.plot(nav_data[6, :], nav_data[5, :], 'r.-')
		plt.plot(interp_arr, r_smooth)
		plt.ylabel('r')
		plt.grid()

		plt.figure()
		plt.subplot(211)
		plt.plot(jet_data[2, :], jet_data[1, :])
		plt.grid()
		plt.ylabel('nozzle')
		plt.subplot(212)
		plt.plot(nav_data[6, :], nav_data[4, :])
		plt.grid()
		plt.ylabel('v')

		plt.figure()
		plt.plot(nav_data[0,:],nav_data[1,:])
		plt.plot(nav_data[0,0],nav_data[1,0],'rx')
		plt.title('XY-plot, starts at the red cross')

		# plt.figure()
		# plt.plot(nav_data[6,:],nav_data[2, :])
		# plt.title('heading angle')
		# plt.grid()
		plt.show()

	### --- save -- does not work any more
	if save:
		##  - interpolate --
		#interpolate to jet time
		data = np.zeros((10,len(jet_data[2, :])))
		data[0, :] = np.interp(jet_data[2,:], nav_data[6, :],u_smooth)
		data[1, :] = np.interp(jet_data[2,:], nav_data[6, :],v_smooth)
		data[2, :] = np.interp(jet_data[2,:], nav_data[6, :],r_smooth)
		data[3, :] = np.interp(jet_data[2,:], nav_data[6, :],du_smo)
		data[4, :] = np.interp(jet_data[2,:], nav_data[6, :],dv_smo)
		data[5, :] = np.interp(jet_data[2,:], nav_data[6, :],dr_smo)

		data[6, :] = jet_data[0, :] # engine 
		data[7, :] = jet_data[1, :] # steering
		data[8, :] = jet_data[3, :] # bucket
		data[9, :] = jet_data[2, :] # jet time


		save_path = '/home/gislehalv/Master/Data/CSV Data From Bags/'
		save_name = save_path + name + '.csv'
		np.savetxt(save_name, data, delimiter = ',')


	return X


# get data
open_bag(bagFile_path_train, plot=True, thr_bucket = False)
open_bag(bagFile_path_test, plot=True, thr_bucket = False)



