import MCF
import custom_plots
import numpy as np
import pickle


### import the prerecorded data and corresponding parameters
f = open('data/simulation_two_periodic_stimuli_1' + '.pckl', 'rb')
t_sim,response_sim,sigma,mu,tau,omega_1,epsilon_1,omega_2,epsilon_2 = pickle.load(f)
f.close()


### MCF Method: compute the response functions (up to 2nd order) and response to sum of cosine signals
response_func = MCF.response_funcs_two_cosine_signals(tau,sigma,mu,omega_1,omega_2,100,100)

response_to_sum,t = MCF.response_two_cosine_signals(
    epsilon_1,omega_1,
    epsilon_2,omega_2,
    response_func,
    t_start=t_sim[0],t_end=t_sim[-1],t_res=500)

# as a reference once can also plot the sum of (2nd order) responses to each individual cosine
# this will visualize the effect on the response of both cosine signals entering the neuron simultanously 
# for the computation of the sum of responses we just set the response function for the cross-effects zero
response_func_individual = (response_func[0],response_func[1],0,0)

sum_of_responses,t = MCF.response_two_cosine_signals(
    epsilon_1,omega_1,
    epsilon_2,omega_2,
    response_func_individual,
    t_start=t_sim[0],t_end=t_sim[-1],t_res=500)


### plot response including the original signal
signal = epsilon_1*np.cos(omega_1*t) + epsilon_2*np.cos(omega_2*t)

custom_plots.plot_response(
    t,signal,
    lines_resp_y = (response_sim,response_to_sum,sum_of_responses),
    lines_resp_x = (t_sim,t,t),
    lines_resp_label = ('Simulation','MCF method: response to sum','MCF method: sum of responses'),
    parameters=(mu,sigma,tau,epsilon_1,omega_1,epsilon_2,omega_2),
    title='Response to two periodic stimuli')