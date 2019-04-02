"""
- clean up eq

- find the impact of the parts

- find new LS pararmerters ?
"""
from sympy import sympify, cos, sin, expand, collect, Lambda, lambdify, symbols
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/gislehalv/Master/scripts')
import my_lib
import math
import numpy as np

# eq = ' 6.682314804253096e-5*delta_n*v + 8.42718325829966e-12*delta_t**2*Abs(cos(r + v) + Abs(delta_t + Abs(delta_t))) - 4.690072989652602e-9*delta_t**2 + 6.090976302400539e-6*delta_t*u + 0.00023111960750050932*r*u - 0.017220470456287523*u - 0.042251018536503204*sin(Abs(r)) + 0.0009319604161203575*sin(Abs(u)) - 0.00782428632744549*cos(u) + 0.02275089408077216*cos(2*v) + 0.0003487568807687741*cos(delta_n*v + 2*r) - 0.0027080761346341495*cos(cos(delta_t + v)) + 0.0004551356960586137*Abs(u)**2 + 0.02540651614422688*Abs(v) - 2.5404073997960238e-9*Abs(delta_t**2*u) + 0.0001753254711668482*Abs(cos(delta_n))'
# expand_str = expand(eq)

# #print(expand_str)


# exit()

exp_str  = '6.682314804253096e-5*delta_n*v + 8.42718325829966e-12*delta_t**2*Abs(cos(r + v) + Abs(delta_t + Abs(delta_t))) - 4.690072989652602e-9*delta_t**2 + 6.090976302400539e-6*delta_t*u + 0.00023111960750050932*r*u - 0.017220470456287523*u - 0.042251018536503204*sin(Abs(r)) + 0.0009319604161203575*sin(Abs(u)) - 0.00782428632744549*cos(u) + 0.02275089408077216*cos(2*v) + 0.0003487568807687741*cos(delta_n*v + 2*r) - 0.0027080761346341495*cos(cos(delta_t + v)) + 0.0004551356960586137*Abs(u)**2 + 0.02540651614422688*Abs(v) - 2.5404073997960238e-9*Abs(delta_t**2*u) + 0.0001753254711668482*Abs(cos(delta_n))'


def expand_function(exp_str):
	u, v, r, delta_t, delta_n = symbols('u v r delta_t delta_n')
	f = lambdify((u,v,r, delta_t, delta_n), exp_str, 'numpy')
	return f

func = expand_function(exp_str)



bag_1 = 'hal_control_2018-12-11-10-53-26_0' #large!
bag_2 = 'hal_control_2018-12-11-11-49-22_0' #similar to bag1 but smaller
bag_3 = 'hal_control_2018-12-11-12-13-58_0' #
bag_4 = 'hal_control_2018-12-11-12-19-11_0'

path = '/home/gislehalv/Master/Data/'


bagFile_path = path + bag_2 + '.bag'


# get data
X = my_lib.open_bag(bagFile_path, plot=False, thr_bucket = False, filter_cutoff = 0.1)
#	X = [u_smooth, v_smooth, r_smooth, du_smo, dv_smo, dr_smo, jet_rpm, nozzle_angle, bucket, interp_arr]

u = X[0]
v = X[1]
r = X[2]
delta_t = X[-4]
delta_n = X[-3]
y = X[3]
time = X[-1]

sol = func(u,v,r,delta_t,delta_n)

#small = small_func(u,v,r,delta_t,delta_n)
#orig_eq = orig_eq_func(u,v,r,delta_t,delta_n)

# tmp = tmp_f(u,v,r,delta_t,delta_n)
# tmp2 =-0.008310182528562431*np.abs(v)-0.007608938618059389*u+0.0007976177486488578*v+0.014028440437266896*np.cos(np.sin(u))+4.4025356707874037e-08*(delta_n + delta_t)*(delta_t + r)+0.019835875070941006*np.sin(np.cos(2*r))-4.290191942426662e-05*delta_t
# plt.figure()
# plt.plot(time, tmp2, 'rx')
# plt.plot(time, tmp,)
# plt.plot(time, y)
# plt.legend(['tmp2', 'tmp', 'GT'])
# plt.show()
# exit()


# mse = math.fsum((orig - small)**2)/len(small)
# print(mse)
# exit()
# plt.figure()
# plt.plot(time, func(u,v,r,delta_t,delta_n))
# plt.plot(time, small_func(u,v,r,delta_t,delta_n))
# plt.plot(time, y)
# plt.grid()
# plt.legend(['orig', 'small', 'GT'])



####---- Find the parts of the eq that have tooo small values
sub_func_list_small = ['6.682314804253096e-5*delta_n*v ',' + 8.42718325829966e-12*delta_t**2*Abs(cos(r + v) + Abs(delta_t + Abs(delta_t))) ',' - 4.690072989652602e-9*delta_t**2 ','+ 6.090976302400539e-6*delta_t*u ','+ 0.00023111960750050932*r*u ','- 0.017220470456287523*u ','- 0.042251018536503204*sin(Abs(r)) ','+ 0.0009319604161203575*sin(Abs(u)) ','- 0.00782428632744549*cos(u) ','+ 0.02275089408077216*cos(2*v) ','+ 0.0003487568807687741*cos(delta_n*v + 2*r) ','- 0.0027080761346341495*cos(cos(delta_t + v)) ','+ 0.0004551356960586137*Abs(u)**2 ','+ 0.02540651614422688*Abs(v)',' - 2.5404073997960238e-9*Abs(delta_t**2*u) ','+ 0.0001753254711668482*Abs(cos(delta_n))']
func_list_rest  = []
for sub_f in sub_func_list_small:

	sub_function = expand_function(sub_f)

	if np.amax(np.abs(sub_function(u,v,r,delta_t,delta_n))) < 1e-3:
		print('removed:', sub_f,'  the largest absolute value were:', np.amax(np.abs(sub_function(u,v,r,delta_t,delta_n))))

	else:
		func_list_rest.append(sub_f)

#print(func_list_rest)
#exit()


#####






#func_list_rest = ['2.1016177152619688e-11*delta_t**3', '- 3.968594986798428e-8*delta_t**2', '- 3.968594986798428e-8*delta_t*v', '- 3.5885398311604876e-5*delta_t' ,' + 0.002687931490990003*r*v',' - 0.010028718337762504*cos(u) ',' + 2.404803597293958e-7*Abs(delta_t)*Abs(delta_n*sin(delta_t) + v*sin(delta_t)) ',' - 0.013751341138746428*Abs(u) ','+ 0.00037713229593790306*Abs(u**2)',' + 9.779294443528126e-5*Abs(delta_t + u**2*cos(u))']

plt.figure()


for i, sub_func in enumerate(func_list_rest):

	sub_function = expand_function(sub_func)
	plt.subplot(len(func_list_rest), 1, i+1)
	new_func = sub_function(u,v,r,delta_t,delta_n)
	plt.plot(time, new_func)
	plt.legend([sub_func])


	if  i == 0:
		tot_func = new_func
	else:
		tot_func = np.add(tot_func, new_func)


plt.xlabel('Time [s]')

plt.figure()
plt.plot(time, tot_func)
plt.plot(time, sol)
plt.plot(time, y)
plt.grid()
plt.legend(['tot_func', 'orig' ,'GT'])




plt.show()