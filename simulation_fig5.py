from simulation import theta_torch
import numpy as np
import time
import pickle
from math import pi,sqrt

T = 110
dt = 0.002
sigma = 1
n_ens = 1000000
t_relax = 10
n_relax = int()
signal = np.zeros(int(T/dt))
noise_ens = np.zeros(n_ens)
phase_ens = np.linspace(0,2*pi,n_ens)

# span tau vector
tau_start = 0.01
tau_end = 10
tau_res = 10
tau_vec = np.zeros(tau_res)
dt_tau = pow(tau_end/tau_start,1/(tau_res-1))
for n in range(tau_res):
    tau_vec[n] = tau_start*pow(dt_tau,n)

# span mu vector
mu_vec = np.array([1.0,3.0,5.0])

stat_firing_rate = np.zeros((len(mu_vec),len(tau_vec[1:-1])))

i = 0

for mu in mu_vec:

    j = -1
    for tau in tau_vec[:-1]: # the first simulation point is just for estimating the the phase distribution
        start = time.time()
        firing_rate,_,noise_ens,phase_ens = theta_torch(signal,dt,mu,sigma,tau,noise_ens,phase_ens) # reuse phase and noise dist for adjascent points
        t_elapsed = time.time()-start
        
        if j >= 0:
            stat_firing_rate[i,j] = np.mean(firing_rate[n_relax:])
            print('\n','firing rate',stat_firing_rate[i,j])
        print(i,j,'time torch', t_elapsed)

        j += 1

    i += 1


name = 'theta_sim_fig5'
f = open(name + '.pckl', 'wb')
pickle.dump([dt,T,t_relax,n_ens,tau_vec[1:-1],sigma,mu_vec,stat_firing_rate[:,:]], f)
f.close()