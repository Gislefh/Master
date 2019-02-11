import rosbag
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import integrate, signal, interpolate


path = '/home/gislehalv/Master/Data/'
name = 'hal_control_2018-12-11-12-19-11_0'
bagFile = path + name + '.bag'
bag = rosbag.Bag(bagFile)
bagContents = bag.read_messages()
if not bagContents:
	print('bag is empty')
	exit()

#save?
save = False


#listOfTopics = []
#for topic, msg, t in bagContents:
#	if topic not in listOfTopics:
#		listOfTopics.append(topic)
#print('TOPICS:')
#print(listOfTopics)


#### -- jet data --
cnt_jet = 0
for subtopic, msg, t in bag.read_messages('/usv/hamjet/status_high'):
	cnt_jet += 1

jet_data  = np.zeros((4,cnt_jet))
i = 0
for subtopic, msg, t in bag.read_messages('/usv/hamjet/status_high'):
	# engines rpm, steering, jet time, 
	jet_data[0, i] = (msg.port_shaft_rpm + msg.stbd_shaft_rpm) / 2
	jet_data[1, i] = (msg.port_steering + msg.stbd_steering) / 2
	jet_data[2, i] = t.to_sec()
	jet_data[3, i] = (msg.port_reverse + msg.stbd_reverse) / 2
	i += 1

# 
jet_data[2, :] = jet_data[2, :] - jet_data[2, 0]


#### -- nav data --
cnt_nav = 0
for subtopic, msg, t in bag.read_messages('/usv/navp_msg'):
	cnt_nav += 1

#print('Available nav_data in /usv/navp_msg:',msg)
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
	nav_data[5, i] = msg.rate.zAngularRateB

	# nav time
	nav_data[6, i] = t.to_sec()
	i += 1

# 
nav_data[6, :] = nav_data[6, :] - nav_data[6, 0] 

####		 --Fiter--
u_smooth = signal.savgol_filter(nav_data[3, :], 501, 3) 
v_smooth = signal.savgol_filter(nav_data[4, :], 501, 3) 
r_smooth = signal.savgol_filter(nav_data[5, :], 101, 3) 



#-- notch filter for yaw
#butter
b, a = signal.butter(3, 0.05)
zi = signal.lfilter_zi(b, a)


avr_step_len = nav_data[6, -1] / len(nav_data[6, :]) #avr. step len
fs = 1/avr_step_len


sp = np.fft.rfft(r_smooth)
freq = np.fft.rfftfreq(len(r_smooth) , avr_step_len)
plt.plot(freq, sp.real)
plt.show()

w0 = 0.01
Q = 1

print(fs)
b, a = signal.iirnotch(w0, Q, fs = fs)
r_notch = signal.lfilter(b, a, r_smooth)

plt.figure()
plt.subplot(311)
plt.plot(nav_data[6, :], nav_data[5, :])
plt.plot(nav_data[6, :], r_smooth)
plt.plot(nav_data[6, :], r_notch)
plt.grid()
plt.legend(['orig signal','original r', 'notch filtered'])
plt.subplot(312)
plt.plot(jet_data[2, :], jet_data[1, :])
plt.grid()
plt.legend(['steering'])
plt.subplot(313)
plt.plot(nav_data[6, :], nav_data[2, :])
plt.legend(['heading'])
plt.show()
exit()




#### 		-- Integrate --
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

### --- save --
if save:
	save_path = '/home/gislehalv/Master/Data/CSV Data From Bags/'
	save_name = save_path + name + '.csv'
	np.savetxt(save_name, data, delimiter = ',')

## data = [u, v, r, du, dv, dr, engine rpm, steering nozzle, bucket, time]



### 		---Plots----
plt.figure()
plt.plot(nav_data[0,:],nav_data[1,:])
plt.plot(nav_data[0,0],nav_data[1,0],'rx')
plt.title('XY-plot, starts at the red cross')

plt.figure()
plt.plot(nav_data[6,:],nav_data[2, :])
plt.title('heading angle')
plt.grid()

plt.figure()
plt.plot(nav_data[6,:],nav_data[3, :])
plt.plot(nav_data[6,:],nav_data[4, :])
plt.plot(nav_data[6,:],nav_data[5, :])
plt.legend(['u','v','r'])
plt.grid()

plt.figure()
plt.plot(jet_data[2, :], jet_data[0, :])
plt.plot(jet_data[2, :], jet_data[1, :])
plt.plot(jet_data[2, :], jet_data[3, :])
plt.legend(['rpm','steering','bucket'])
plt.grid()

plt.figure()
plt.subplot(311)
plt.plot(nav_data[6, :], du)
plt.plot(nav_data[6, :], du_smo)
plt.subplot(312)
plt.plot(nav_data[6, :],dv)
plt.plot(nav_data[6, :],dv_smo)
plt.subplot(313)
plt.plot(nav_data[6, :],dr)
plt.plot(nav_data[6, :],dr_smo)
plt.grid()

plt.figure()
plt.subplot(311)
plt.plot(nav_data[6, :], nav_data[3, :])
plt.plot(nav_data[6, :], u_smooth)
plt.ylabel('u')
plt.grid()
plt.subplot(312)
plt.plot(nav_data[6, :], nav_data[4, :])
plt.plot(nav_data[6, :], v_smooth)
plt.ylabel('v')
plt.grid()
plt.subplot(313)
plt.plot(nav_data[6, :], nav_data[5, :])
plt.plot(nav_data[6, :], r_smooth)
plt.ylabel('r')
plt.grid()

plt.show()







#timesteps

# cnt = 0
# for i in range(len(nav_data[6, :])):
# 	ts = nav_data[6, 1] - nav_data[6, 0]
# 	if not (nav_data[6, i+1] - nav_data[6, i]) == ts:
# 		cnt +=1
# 		print(cnt, (nav_data[6, i+1] - nav_data[6, i]), ts)

# cnt = 0
# for i in range(len(jet_data[2, :])):
# 	ts = jet_data[2, 1] - jet_data[2, 0]
# 	if not (jet_data[2, i+1] - jet_data[2, i]) == ts:
# 		cnt +=1
# 		print(cnt, (jet_data[2, i+1] - jet_data[2, i]), ts)

