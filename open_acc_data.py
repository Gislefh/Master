"""

open acc data 

"""
import numpy as np
import matplotlib.pyplot as plt
import rosbag
# path = '/home/gislehalv/Master/Data/NavData/'
# file0 = 'NavigationSolutionData-0-0000.txt'
# file1 = 'NavigationSolutionData-0-0001.txt'
# file2 = 'NavigationSolutionData-0-0002.txt'
# file3 = 'NavigationSolutionData-0-0003.txt'
# file4 = 'NavigationSolutionData-0-0004.txt'
# file5 = 'NavigationSolutionData-0-0005.txt'
# file6 = 'NavigationSolutionData-0-0006.txt'
# file7 = 'NavigationSolutionData-0-0007.txt'
# file8 = 'NavigationSolutionData-0-0008.txt'
# file9 = 'NavigationSolutionData-0-0009.txt'
# file10 = 'NavigationSolutionData-0-0010.txt'
# file11 = 'NavigationSolutionData-0-0011.txt'
# file12 = 'NavigationSolutionData-0-0012.txt'
# file13 = 'NavigationSolutionData-0-0013.txt'
# file14 = 'NavigationSolutionData-0-0014.txt'
# file15 = 'NavigationSolutionData-0-0015.txt'
# file16 = 'NavigationSolutionData-0-0016.txt'

# file_list = [file0, file1, file2, file3, file4, file5, file6, file7, file8, file9, file10, file11, file12, file13, file14, file15, file16]

# xacc = []
# lat = []
# long_ = []
# valid_time = []
# xvel = []
# for file in file_list:
# 	file_obj = open(path + file, 'r')



# 	i = 0
# 	for line in file_obj:

# 		if i > 40:
# 			xvel.append(float(line.split(',')[6]))
# 			xacc.append(float(line.split(',')[15]))
# 			valid_time.append(float(line.split(',')[30]) *1000) # to Nsec
# 			lat.append(float(line.split(',')[0]))
# 			long_.append(float(line.split(',')[1]))

# 		i = i+1

# # plt.figure()
# # plt.plot(lat, long_, 'b.')
# # plt.title('lat_long')


# # plt.figure()
# # plt.plot(valid_time, xacc, )





# # plt.show()

# # exit()


# ###rosbag
# import rosbag


path = '/home/gislehalv/Master/Data/'
bag_1 = 'hal_control_2018-12-11-10-53-26_0' #large!
bag_2 = 'hal_control_2018-12-11-11-49-22_0' #similar to bag1 but smaller
bag_3 = 'hal_control_2018-12-11-12-13-58_0' #
bag_4 = 'hal_control_2018-12-11-12-19-11_0'
bag_path = path + bag_3 + '.bag'

# bag = rosbag.Bag(bag_path)
# bagContents = bag.read_messages()

# #### -- jet data --
# cnt_jet = 0
# for subtopic, msg, t in bag.read_messages('/usv/hamjet/status_high'):
# 	cnt_jet += 1
# test_time = []
# jet_data  = np.zeros((4,cnt_jet))
# i = 0
# for subtopic, msg, t in bag.read_messages('/usv/hamjet/status_high'):
# 	# engines rpm, steering, jet time, bucket
# 	jet_data[0, i] = (msg.port_shaft_rpm + msg.stbd_shaft_rpm) / 2
# 	#jet_data[1, i] = (msg.port_steering + msg.stbd_steering) / 2
# 	jet_data[2, i] = float(str(t))#.to_sec()
# 	test_time.append(t.to_sec())
# 	#exit()
# 	jet_data[3, i] = (msg.port_reverse + msg.stbd_reverse) / 2
# 	i += 1



# #### -- nav data --
# cnt_nav = 0
# for subtopic, msg, t in bag.read_messages('/usv/navp_msg'):
# 	cnt_nav += 1

# nav_data = np.zeros((7,cnt_nav))
# i = 0




# for subtopic, msg, t in bag.read_messages('/usv/navp_msg'):
# 	#surge, sway, yaw
# 	#nav_data[0, i] = msg.pose.latitude
# 	#nav_data[1, i] = msg.pose.longitude
# 	#nav_data[2, i] = msg.pose.heading

# 	#surge, sway, yaw - Rate
# 	nav_data[3, i] = msg.vel.xVelocityB
# 	#nav_data[4, i] = msg.vel.yVelocityB
# 	#nav_data[5, i] = msg.rate.zAngularRateB #prob in deg/s

# 	# nav time
# 	nav_data[6, i] = float(str(t)) #.to_sec()
	
# 	i += 1

# #set initial time to zero
# #nav_data[6, :] = nav_data[6, :] - nav_data[6, 0] 




# # print(jet_data[2,0], jet_data[2,-1])
# # print(test_time[0], test_time[-1])
# # exit()

# #--set init time to zero
# nav_data[6, :] = np.divide(np.subtract(nav_data[6, :], valid_time[0]), 1e9) # to sec
# jet_data[2, :]  =  np.divide(np.subtract(jet_data[2,:], valid_time[0]), 1e9)# to sec
# valid_time = np.divide(np.subtract(valid_time, valid_time[0]), 1e9) # to sec


# # 7.8 sec diff

# nav_data[6, :] = nav_data[6, :] + 7.8
# jet_data[2, :] = jet_data[2, :] + 7.8


# plt.figure()
# plt.subplot(211)
# plt.plot(valid_time, xvel)
# plt.plot(nav_data[6] ,nav_data[3])
# plt.subplot(212)
# plt.plot(jet_data[2], jet_data[0])
# #plt.plot(valid_time, [0] * valid_time)

# plt.legend(['.txt file','rosbag' ])
# plt.show()




def acc_data(path_to_bag):
	path = '/home/gislehalv/Master/Data/NavData/'
	file0 = 'NavigationSolutionData-0-0000.txt'
	file1 = 'NavigationSolutionData-0-0001.txt'
	file2 = 'NavigationSolutionData-0-0002.txt'
	file3 = 'NavigationSolutionData-0-0003.txt'
	file4 = 'NavigationSolutionData-0-0004.txt'
	file5 = 'NavigationSolutionData-0-0005.txt'
	file6 = 'NavigationSolutionData-0-0006.txt'
	file7 = 'NavigationSolutionData-0-0007.txt'
	file8 = 'NavigationSolutionData-0-0008.txt'
	file9 = 'NavigationSolutionData-0-0009.txt'
	file10 = 'NavigationSolutionData-0-0010.txt'
	file11 = 'NavigationSolutionData-0-0011.txt'
	file12 = 'NavigationSolutionData-0-0012.txt'
	file13 = 'NavigationSolutionData-0-0013.txt'
	file14 = 'NavigationSolutionData-0-0014.txt'
	file15 = 'NavigationSolutionData-0-0015.txt'
	file16 = 'NavigationSolutionData-0-0016.txt'

	file_list = [file0, file1, file2, file3, file4, file5, file6, file7, file8, file9, file10, file11, file12, file13, file14, file15, file16]

	xacc = []
	yacc = []
	zacc = []
	#lat = []
	#long_ = []
	valid_time = []
	#xvel = []
	for file in file_list:
		file_obj = open(path + file, 'r')

		i = 0
		for line in file_obj:

			if i > 40:
				#xvel.append(float(line.split(',')[6]))
				xacc.append(float(line.split(',')[15]))
				yacc.append(float(line.split(',')[16]))
				zacc.append(float(line.split(',')[17]))
				valid_time.append(float(line.split(',')[30]) *1000) # to Nsec
				#lat.append(float(line.split(',')[0]))
				#long_.append(float(line.split(',')[1]))

			i = i+1


	bag = rosbag.Bag(path_to_bag)
	bagContents = bag.read_messages()

	# #### -- jet data --
	# jet_data  = [] 
	# for i, subtopic, msg, t in enumerate(bag.read_messages('/usv/hamjet/status_high')):
	# 	jet_data[2, i].append(float(str(t)))
	
	#### -- nav data --
	# cnt_jet = 0
	# for subtopic, msg, t in bag.read_messages('/usv/hamjet/status_high'):
	# 	cnt_jet += 1
	# test_time = []
	# jet_data  = np.zeros((4,cnt_jet))
	# i = 0
	nav_data = []
	for subtopic, msg, t in bag.read_messages('/usv/navp_msg'):
		nav_data.append(float(str(t)))
		
	#--set init time to zero
	nav_data = np.divide(np.subtract(nav_data, valid_time[0]), 1e9) # to sec
	#jet_data[2, :]  =  np.divide(np.subtract(jet_data[2,:], valid_time[0]), 1e9)# to sec
	valid_time = np.divide(np.subtract(valid_time, valid_time[0]), 1e9) # to sec


	# about 7.8 sec diff
	nav_data = np.add(nav_data, 7.8)
	
	

	start = np.argmin((valid_time - nav_data[0])**2)
	stop = np.argmin((valid_time - nav_data[-1])**2)
	
	return xacc[start:stop], yacc[start:stop], zacc[start:stop], valid_time[start:stop]
	

		
acc_data(bag_path)