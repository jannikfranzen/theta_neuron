import numpy as np
from math import sqrt,pi
import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def theta(signal,dt,mu,sigma,tau,noise_ens,phase_ens):

    # setup parameters
    N_ens = len(phase_ens)
    N = len(signal)
    period = 2*pi
    t = 0

    # precompute constants for noise generation
    A = 1-dt/tau 
    B = sqrt(dt*2*sigma*sigma/tau)

    # initialization 
    mean_phase_vel = np.zeros(N)
    firing_rate = np.zeros(N)


    for i in range(N):
        
        # generate noise
        noise_ens = A*noise_ens+np.random.normal(0,B,N_ens)

        # update the phase
        cos_phase = np.cos(phase_ens)
        dPhaseVel = 1+cos_phase + (1-cos_phase)*(noise_ens+mu+signal[i])
        phase_ens = (phase_ens + dPhaseVel*dt) 

        # compute all spiking neurons, and reset the phase of spiking neurons
        spikes = phase_ens/period
        spikes = spikes.astype(int)
        phase_ens = phase_ens % period

        # compute the firing rate (and mean phase velocity, which is an alternative way of computing the firing rate in the stationary case)
        mean_phase_vel[i] = np.mean(dPhaseVel)
        firing_rate[i] = np.mean(spikes)/dt

        t += dt

    return firing_rate,mean_phase_vel/period,noise_ens,phase_ens



def theta_torch(signal,dt,mu,sigma,tau,noise_ens,phase_ens):

    # setup parameters
    N_ens = len(phase_ens)
    N = len(signal)
    period = 2*pi
    t = 0

    # precompute constants for noise generation
    A = 1-dt/tau 
    B = sqrt(dt*2*sigma*sigma/tau)

    # initialization 
    mean_phase_vel = torch.zeros(N).to(DEVICE)
    firing_rate = torch.zeros(N).to(DEVICE)
    noise_ens_torch = torch.from_numpy(noise_ens).to(DEVICE)
    phase_ens_torch = torch.from_numpy(phase_ens).to(DEVICE)


    for i in range(N):
        
        # generate noise
        noise_ens_torch = A*noise_ens_torch+B*torch.randn(N_ens,device=DEVICE)

        # update the phase
        cos_phase = torch.cos(phase_ens_torch)
        dPhaseVel = 1+cos_phase + (1-cos_phase)*(noise_ens_torch+mu+signal[i])
        phase_ens_torch = (phase_ens_torch + dPhaseVel*dt) 

        # compute all spiking neurons, and reset the phase of spiking neurons
        spikes = torch.floor(phase_ens_torch/period)
        phase_ens_torch = phase_ens_torch % period

        # compute the firing rate (and mean phase velocity, which is an alternative way of computing the firing rate in the stationary case)
        mean_phase_vel[i] = torch.mean(dPhaseVel)
        firing_rate[i] = torch.mean(spikes)/dt

        t += dt

    return firing_rate.cpu().detach().numpy(), mean_phase_vel.cpu().detach().numpy()/period, noise_ens_torch.cpu().detach().numpy(), phase_ens_torch.cpu().detach().numpy()


import time
import matplotlib.pyplot as plt

def test(n_ens=1000000):

    T = 10
    dt = 0.05
    mu = 1
    tau = 1
    sigma = 1

    signal = np.zeros(int(T/dt))
    noise_ens = np.zeros(n_ens)
    phase_ens = np.linspace(0,2*pi,n_ens)

    # Numpy implentation
    start = time.time()
    firing_rate,_,_,_ = theta(signal,dt,mu,sigma,tau,noise_ens,phase_ens)
    t_elapsed = time.time()-start
    print('time', t_elapsed)
    plt.plot(firing_rate)

    # PyTorch cuda implementation
    start = time.time()
    firing_rate,_,_,_ = theta_torch(signal,dt,mu,sigma,tau,noise_ens,phase_ens)
    t_elapsed = time.time()-start
    print('time torch', t_elapsed)
    plt.plot(firing_rate)
    plt.show()


if __name__ == "__main__":
    
    test()