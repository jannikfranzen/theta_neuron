import MCF
import custom_plots
import numpy as np
import pickle


### import the prerecorded data and corresponding parameters
f = open('data/simulation_stationary_firing_1' + '.pckl', 'rb')
tau_sim,firing_rate_sim,sigma,mu = pickle.load(f)
f.close()


### MCF method: compute the firing rate for multiple correlation times tau
res = 25
tau = np.logspace(np.log10(0.02),np.log10(10),res)
firing_rate = np.zeros(res)

for i in range(res): 
    firing_rate[i] = MCF.stationary_firing_rate(tau[i],sigma,mu,n_max = 100,p_max = 100)
    print('finished computing datapoint',i+1,'/',res)


### plot the firing rate
custom_plots.plot_stationary_rate(tau,firing_rate,tau_sim,firing_rate_sim,parameters=(mu,sigma),title='Stationary firing rate')