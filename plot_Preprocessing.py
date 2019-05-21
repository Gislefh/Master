import sys
sys.path.insert(0, '/home/gislehalv/Master/scripts')
import my_lib

import matplotlib.pyplot as plt



#### no filter





bag_1 = 'hal_control_2018-12-11-10-53-26_0' #large!
bag_2 = 'hal_control_2018-12-11-11-49-22_0' #similar to bag1 but smaller
bag_3 = 'hal_control_2018-12-11-12-13-58_0' #
bag_4 = 'hal_control_2018-12-11-12-19-11_0'

path = '/home/gislehalv/Master/Data/'
bag_path = path + bag_4 + '.bag'


if 1:
	X = my_lib.open_bag(bag_path, plot=True, thr_bucket = False, filter_cutoff = 0.1)
	exit()




cut_list = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
#cut_list_str = ['0.025', '0.05', '0.75', '0.1', '0.15', '0.2', '0.3']
cut_list_str = ['0.3', '0.2', '0.15', '0.1', '0.075', '0.05', '0.025']
for i,cutoff in enumerate(reversed(cut_list)):
	print(cutoff)
	X = my_lib.open_bag(bag_path, plot = False, thr_bucket = False, filter_cutoff = cutoff)
	

	#plot
	plt.figure(1)
	plt.subplot(311)
	plt.plot(X[-1], X[3], color = (1, i/len(cut_list),0))
	plt.ylabel('du')
	plt.legend(cut_list_str)
	plt.grid()

	plt.subplot(312)
	plt.plot(X[-1], X[4], color = (1, i/len(cut_list),0))
	plt.ylabel('dv')
	plt.legend(cut_list_str)
	plt.grid()

	plt.subplot(313)
	plt.plot(X[-1], X[5], color = (1, i/len(cut_list),0))
	plt.ylabel('dr')
	plt.legend(cut_list_str)
	plt.xlabel('Time [s]')
	plt.grid()


	plt.figure(2)
	plt.subplot(311)
	plt.plot(X[-1], X[0], color = (1, i/len(cut_list),0))
	plt.ylabel('u')
	plt.grid()
	plt.legend(cut_list_str)
	
	plt.subplot(312)
	plt.plot(X[-1], X[1], color = (1, i/len(cut_list),0))
	plt.ylabel('v')
	plt.grid()
	plt.legend(cut_list_str)
	

	plt.subplot(313)
	plt.plot(X[-1], X[2], color = (1, i/len(cut_list),0))
	plt.grid()
	plt.ylabel('r')
	plt.legend(cut_list_str)
	plt.xlabel('Time [s]')
	
	plt.figure(3)
	plt.subplot(211)
	plt.plot(X[-1], X[3], color = (1, i/len(cut_list),0))
	plt.ylabel('du')
	plt.grid()
	plt.legend(cut_list_str)

	plt.figure(4)
	plt.subplot(311)
	plt.plot(X[-1], X[4], color = (1, i/len(cut_list),0))
	plt.ylabel('dv')
	plt.grid()
	plt.legend(cut_list_str)

	plt.subplot(312)
	plt.plot(X[-1], X[5], color = (1, i/len(cut_list),0))
	plt.ylabel('dr')
	plt.grid()
	plt.legend(cut_list_str)

	plt.figure(5)
	plt.plot(X[-1], X[5], color = (1, i/len(cut_list),0))
	plt.ylabel('dr')
	plt.legend(cut_list_str)
	plt.xlabel('Time [s]')
	plt.grid()


plt.figure(4)
plt.subplot(313)
plt.plot(X[-1], X[-3])
plt.ylabel('Nozzle Angle [deg]')
plt.grid()


plt.figure(3)
plt.subplot(212)
plt.plot(X[-1], X[-4])
plt.ylabel('RPM')
plt.grid()


plt.show()

