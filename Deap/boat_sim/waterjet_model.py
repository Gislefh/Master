#water jet model


import numpy as np
import matplotlib.pyplot as plt




def jet_model(nu, jet_rpm, delta_nozzle):
	lever_CGtowj_port = [-3.82, -0.475]
	lever_CGtowj_stb = [-3.82, -0.475]

	#max min rpm
	rpm_min_max = [0, 2000]

	if jet_rpm > rpm_min_max[1]:
		jet_rpm = rpm_min_max[1]
	elif jet_rpm < rpm_min_max[0]:
		jet_rpm = rpm_min_max[0]

	#rate limiter rpm
	rpm_slew_rate = [2000, -2000]
	rate = (jet_now - jet_prev)/(t_now-t_prev)
	if rate > rpm_slew_rate[0]:
		new_rpm = (t_now-t_prev)*rpm_slew_rate[0] + jet_prev
	elif rate < rpm_slew_rate[1]:
		new_rpm = (t_now-t_prev)*rpm_slew_rate[1] + jet_prev

	#rate limiter nozzle
	nozzle_slew_rate = [1.3464, -1.3464]
	rate = (noz_now - noz_prev)/(t_now-t_prev)
	if rate > nozzle_slew_rate[0]:
		new_rpm = (t_now-t_prev)*nozzle_slew_rate[0] + jet_prev
	elif rate < nozzle_slew_rate[1]:
		new_rpm = (t_now-t_prev)*nozzle_slew_rate[1] + jet_prev



	#rpm2thrust
	u = nu[0] * 1.94384 # knots 
	a0 = 6244.15
	a1 = -178.46
	a2 = 0.881043
	thrust_unscaled = a0 + a1*speed + a2*speed**2

	r0 = 85.8316
	r1 = -1.7935
	r2 = 0.00533
	rmp_scale = 1/4530*(r0 + r1*rpm + r2 * rmp **2)

	thrust = rpm_scale * thrust_unscaled


	#waterjet port
	#force
	Fx = thrust*np.cos(delta_nozzle)
	Fy = thrust*np.sin(delta_nozzle)
	#moment
	Nz = (lever_CGtowj[0]*Fy)- (lever_CGtowj[1]*Fx)

	tau_b_port = [Fx, Fy, Nz]
	tau_b_stb = [Fx, Fy, Nz]

	tau_b = np.add(tau_b_port, tau_b_stb)


