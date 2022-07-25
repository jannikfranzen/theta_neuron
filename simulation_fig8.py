from tokenize import PlainToken
from simulation import theta_torch
import numpy as np
import time
import pickle
from math import pi,sqrt
import matplotlib.pyplot as plt

T = 20
dt = 0.005
n_ens = 10000000
t_vec = np.arange(0,T,dt)
noise_ens = np.zeros(n_ens)
phase_ens = np.zeros(n_ens)#np.linspace(0,2*pi,n_ens)

# model parameter
mu = 0.5
sigma = 1
tau = 1
epsilon = 0.1
omega = 2

signal = epsilon*np.cos(omega*t_vec)

firing_rate,_,noise_ens,phase_ens = theta_torch(signal,dt,mu,sigma,tau,noise_ens,phase_ens) 

name = 'theta_sim_fig8'
f = open(name + '.pckl', 'wb')
pickle.dump([dt,T,n_ens,tau,sigma,mu,epsilon,omega,firing_rate], f)
f.close()

plt.plot(firing_rate)
plt.show()
