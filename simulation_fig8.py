from simulation import theta_torch
import numpy as np
import time
import pickle
from math import pi,sqrt

T = 20
dt = 0.002
sigma = 1
n_ens = 1000000
n_relax = int()
t_vec = np.zeros(int(T/dt))
noise_ens = np.zeros(n_ens)
phase_ens = np.linspace(0,2*pi,n_ens)

# model parameter
mu = 1
sigma = 1
tau = 1
epsilon = 0.1
omega = 2

signal = epsilon*np.cos(omega*t_vec)

firing_rate,_,noise_ens,phase_ens = theta_torch(signal,dt,mu,sigma,tau,noise_ens,phase_ens) # reuse phase and noise dist for adjascent points




name = 'theta_sim_fig8'
f = open(name + '.pckl', 'wb')
pickle.dump([dt,T,n_ens,tau,sigma,mu,firing_rate], f)
f.close()