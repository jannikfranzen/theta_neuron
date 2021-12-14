import MCF
import custom_plots
import numpy as np
import pickle


### import the prerecorded data and corresponding parameters
f = open('data/simulation_periodic_stimulus_3' + '.pckl', 'rb')
t_sim,response_sim,sigma,mu,tau,omega,epsilon = pickle.load(f)
f.close()


### MCF method: compute the response functions and response
response_func = MCF.response_funcs_cosine_signal(tau, sigma,mu,omega,l_max=5, n_max=100, p_max=100)

period = 2*np.pi/omega
t_start = 0
t_end = 2*period
t_res = 500

response,t = MCF.response_cosine_signal(epsilon,omega,response_func,t_start,t_end,t_res)


### plot response including the original signal
signal = epsilon*np.cos(omega*t)

custom_plots.plot_response(
    t,signal,
    lines_resp_y = (response_sim,response),
    lines_resp_x = (t_sim,t),
    lines_resp_label = ('Simulation','MCF method'),
    parameters=(mu,sigma,tau,epsilon,omega),
    title='Response to periodic stimulus')