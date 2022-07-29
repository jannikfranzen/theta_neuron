from simulation import theta_torch
import numpy as np
from math import pi
import custom_plots
from tqdm import tqdm



# set model parameter 
sigma = 1
mu = -.5
tau = np.logspace(-1.2,1)

# set the simulation parameters: time window T, step width dt, ensemble size n_ens
T = 20        
dt = 0.02
n_ens = 50000

# choose the time window, which should not be considered due to the transient relaxation 
t_relax = 5
n_relax = int(t_relax/dt)

# no signal is present:
t_vec = np.arange(0,T,dt)
signal = np.zeros(t_vec.shape)



# STEP 1: compute an initial noise and phase distribution
noise_ens = np.zeros(n_ens)
phase_ens = np.linspace(-pi,pi,n_ens)

# let the ensemble relax in its stationary state for the first data point 
_,noise_ens,phase_ens = theta_torch(signal,dt,mu,sigma,tau[0],noise_ens,phase_ens) 



# STEP 2: compute the firing rate
stat_firing_rate = np.zeros(tau.shape)
j = 0

for i in tqdm(range(len(tau))): 
    firing_rate,noise_ens,phase_ens = theta_torch(signal,dt,mu,sigma,tau[i],noise_ens,phase_ens) 
    # note: we reuse the noise and phase ensemble at the end of our timewindow T
    #       as initial values for the adjascent data point  

    stat_firing_rate[i] = np.mean(firing_rate[n_relax:])



# Plot the results
custom_plots.plot_stat_simulation(tau,stat_firing_rate,parameters=(mu,sigma),title='Simulation STAT. FIRING RATE')