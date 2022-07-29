import numpy as np
from math import sqrt,pi
import torch




def theta(signal,dt,mu,sigma,tau,noise_ens,phase_ens):

    # setup parameters
    N_ens = len(phase_ens)
    N = len(signal)
    period = 2*pi

    # shift the phase from [-pi,pi] to [0,2*pi] for computational convenience
    # note: sign before cos() changes
    phase_ens = phase_ens + pi 

    # precompute constants for noise generation
    A = 1-dt/tau 
    B = sqrt(dt*2*sigma*sigma/tau)

    # initialization 
    firing_rate = np.zeros(N)


    for i in range(N):
        
        # generate noise
        noise_ens = A*noise_ens+np.random.normal(0,B,N_ens)

        # update the phase
        cos_phase = np.cos(phase_ens)
        phase_ens = phase_ens + (1+cos_phase + (1-cos_phase)*(noise_ens+mu+signal[i]))*dt 

        # compute all spiking neurons, and reset the phase of spiking neurons
        spikes = phase_ens/period
        spikes = spikes.astype(int)
        phase_ens = phase_ens % period

        # compute the firing rate
        firing_rate[i] = np.mean(spikes)/dt


    return firing_rate,noise_ens,phase_ens




def theta_torch(signal,dt,mu,sigma,tau,noise_ens,phase_ens):

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    # setup parameters 
    N_ens = len(phase_ens)
    N = len(signal)
    period = 2*pi

    # precompute constants for noise generation
    A = 1-dt/tau 
    B = sqrt(dt*2*sigma*sigma/tau)

    # initialization 
    firing_rate = torch.zeros(N).to(DEVICE)
    noise_ens_torch = torch.from_numpy(noise_ens).to(DEVICE)
    phase_ens_torch = torch.from_numpy(phase_ens+pi).to(DEVICE) # shift phase as in theta()


    for i in range(N):
        
        # generate noise
        noise_ens_torch = A*noise_ens_torch + B*torch.randn(N_ens,device=DEVICE)

        # update the phase
        cos_phase = torch.cos(phase_ens_torch)
        phase_ens_torch += (1+cos_phase + (1-cos_phase)*(noise_ens_torch+mu+signal[i]))*dt 

        # compute all spiking neurons, and reset the phase of spiking neurons
        spikes = torch.floor(phase_ens_torch/period)
        phase_ens_torch = phase_ens_torch % period

        # compute the firing rate
        firing_rate[i] = torch.mean(spikes)/dt


    return firing_rate.cpu().detach().numpy(), noise_ens_torch.cpu().detach().numpy(), phase_ens_torch.cpu().detach().numpy()