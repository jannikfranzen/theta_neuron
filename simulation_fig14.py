from tokenize import PlainToken
from simulation import theta_torch
import numpy as np
import time
import pickle
from math import pi,sqrt
import matplotlib.pyplot as plt

T = 200
dt = 0.005
n_ens = 500000
t_vec = np.arange(0,T,dt)
noise_ens = np.zeros(n_ens)
phase_ens = np.linspace(0,2*pi,n_ens)

# model parameter
mu = 1
sigma = 1
tau = 0.05
epsilon_1 = 0.3
omega_1 = 1
epsilon_2 = 0.1
omega_2 = 1.5

signal = epsilon_1*np.cos(omega_1*t_vec)+epsilon_2*np.cos(omega_2*t_vec)

firing_rate,_,noise_ens,phase_ens = theta_torch(signal[::-1],dt,mu,sigma,tau,noise_ens,phase_ens)
firing_rate,_,noise_ens,phase_ens = theta_torch(signal,dt,mu,sigma,tau,noise_ens,phase_ens) 

name = 'theta_sim_fig14C'
f = open(name + '.pckl', 'wb')
pickle.dump([dt,T,n_ens,tau,sigma,mu,epsilon_1,omega_1,epsilon_2,omega_2,firing_rate], f)
f.close()

plt.plot(firing_rate)
plt.show()
