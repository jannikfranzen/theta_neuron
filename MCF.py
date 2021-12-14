'''
    This file contains all the functions needed to perform the MCF method 
    in order to compute the stationary firing rate as well es the response
    to one and to periodic stimuli 
'''

import numpy as np
from math import sqrt,pi




######################################
########## STATIONARY STATE ##########
######################################


def stationary_firing_rate(tau,sigma,mu,n_max = 100,p_max = 100,coeff = False):
    '''
        Computes stationary firing rate of the theta neuron 
        driven by correlated noise 
        with matrix-continued fraction (MCF) method \n

        Keyword arguments:\\
            tau -- correlation time \\
            sigma -- standard deviation \\ 
            mu -- mean input \\
            n_max,p_max -- number of fourier modes, hermite functions included \\
            coeff -- True: computes and store expansion coefficients
    '''
        
    ### 1: INITIALIZATION AND PRECOMPUTATIONS ###
    A = np.zeros((p_max,p_max),dtype=np.complex_)
    B = np.zeros((p_max,p_max),dtype=np.complex_)
    S = np.zeros((p_max,p_max),dtype=np.complex_)
    I = np.identity(p_max)

    for p in range(p_max):
        for q in range(p_max):

            if p == q:
                A[p,q] = q/(tau)*1j
                B[p,q] = (1-mu)/2

            if p+1 == q:
                B[p,q] = -sigma/2*sqrt(q)
            
            if p-1 == q:
                B[p,q] = -sigma/2*sqrt(q+1)

    # matrices for computing K_n = M1/n + M2
    M1 = -np.dot(np.linalg.inv(B),A)
    M2 = 2*(np.linalg.inv(B)-I)
      
    # Setting up memory for saving the transition matrices S
    if coeff == True: 
        S_memory = np.zeros((n_max+1,p_max,p_max),dtype=np.complex_)


    ### 2: COMPUTE TRANSITION MATRICES S WITH MCF ###
    for n in range(n_max,0,-1):
        S = np.linalg.inv(M1/n + M2 - S)

        if coeff == True: 
            S_memory[n-1,:,:] = S


    ### 3: COMPUTE STATIONARY FIRING RATE ###
    c11 = S[1,0]
    c01 = S[0,0]

    firing_rate = ((1+mu) - (1-mu)*np.real(c01) + sigma*np.real(c11))/(2*pi)
      

    # if needed (e.g. for calculating the phase distribution) the expansion coefficients c_{n,p} can be computed
    # Note: Because of the symmetry property c_{-n,p}* = c_{n,p}, only coefficients with n = 0,1,...,n_max 
    # will be calculated and stored in c
    if coeff == True:

        c = np.zeros((n_max+1,p_max),dtype=np.complex_)
        c[0,0] = 1

        for n in range(n_max):

            c[n+1,:] = np.dot(S_memory[n,:,:],c[n,:])

        return firing_rate,c

    return firing_rate



#######################################
########## PERIODIC STIMULUS ##########
#######################################


def single_response_func_cosine_signal(l,k,n_max,p_max,tau,sigma,mu,omega,c_P_prevsum,c_N_prevsum):
    '''
        Computes a single response function r_{l,k} of the theta neuron 
        driven by correlated noise and a cosine stimulus 
        with matrix continued fraction (MCF) method \n

        returns:\\
            r_lk -- single response function r_{l,k}\\
            c_P -- computed expansion coefficients for n = 0,1,2,...\\
            c_N -- computed expansion coefficients for n = 0,-1,-2,... \n

        Keyword arguments:\\
            l -- corresponds to terms multiplied by epsilon^l in pertubation series (epsilon is signal amplitude) \\
            k -- corresponds to k-th Fourier mode (Fourier expansion in time) \\
            tau -- correlation time \\
            sigma -- standard deviation \\ 
            mu -- mean input \\
            n_max,p_max -- number of Fourier modes, Hermite functions included \\
            c_P_prevsum,c_N_prevsum -- sum of previousely computed coefficients for l-1,k+-1; 
            '_P' denotes all coeff with n = 0,1,2,... and '_N' with n = 0,-1,-2,... 
    '''
    
    ### 1: INITIALIZATION AND PRECOMPUTATIONS ###
    A = np.zeros((p_max,p_max),dtype=np.complex_)
    B = np.zeros((p_max,p_max),dtype=np.complex_)
    I = np.identity(p_max)

    for p in range(p_max):
        for q in range(p_max):

            if p == q:
                A[p,q] = q/(tau)*1j
                B[p,q] = (1-mu)/2

            if p+1 == q:
                B[p,q] = -sigma/2*sqrt(q)
            
            if p-1 == q:
                B[p,q] = -sigma/2*sqrt(q+1)

    Binv = np.linalg.inv(B)
    
    # for the computation of K_n = M1/n + M2
    M1 = -np.dot(Binv,(A+k*omega*I))
    M2 = 2*(Binv-I)

    # initialize transition matrices S,SR and vectors d,dR
    S = np.zeros((n_max+1,p_max,p_max),dtype=np.complex_)
    SR = np.zeros((n_max+1,p_max,p_max),dtype=np.complex_)

    d = np.zeros((n_max+1,p_max),dtype=np.complex_)
    dR = np.zeros((n_max+1,p_max),dtype=np.complex_)

    
    ### 2: COMPUTE TRANSITION MATRICES S,SR AND VECTORS d,dR WITH MCF ###
    for n in range(n_max,0,-1):
        S[n-1] = np.linalg.inv(M2 + M1/n - S[n])

        c_tilde = -np.dot(Binv,c_P_prevsum[n]/2+(c_P_prevsum[n-1]+c_P_prevsum[n+1])/4)
        d[n-1] = np.dot(S[n-1],c_tilde+d[n]) 

        # for negative indices, i.e. -|n|
        SR[n-1] = np.linalg.inv(M2 - M1/n - SR[n])

        c_tilde = -np.dot(Binv,c_N_prevsum[n]/2+(c_N_prevsum[n-1]+c_N_prevsum[n+1])/4)
        dR[n-1] = np.dot(SR[n-1],c_tilde+dR[n])
 
    
    ### 3: COMPUTE EXPANSION COEFFICIENTS ###
    c_P = np.zeros((n_max+1,p_max),dtype=np.complex_)
    c_N = np.zeros((n_max+1,p_max),dtype=np.complex_)

    if k == 0 and l == 0:
        c_P[0,0] = 1
        c_N[0,0] = 1

    for n in range(n_max):
        
        # iterate upwards
        c_P[n+1] = np.dot(S[n],c_P[n])+d[n]

        # iterate downwards
        c_N[n+1] = np.dot(SR[n],c_N[n])+dR[n]


    ### 4: COMPUTE RESPONSE FUNCTION r_lk ###
    r_lk = c_P[0,0]/pi

    for n in range(1,n_max):
        r_lk += pow(-1,n)*(c_P[n,0]+c_N[n,0])/pi 
    
    if k != 0:
        r_lk = 2*r_lk

    return r_lk,c_P,c_N




def response_funcs_cosine_signal(tau,sigma,mu,omega,l_max = 5,n_max = 100,p_max = 100):
    '''
        Computes ALL the response function r_{l,k} up to the order l_max
        of the theta neuron driven by correlated noise and a cosine stimulus 
        with the matrix continued fraction (MCF) method \n

        Keyword arguments of stochastic system: \\
            tau -- correlation time \\
            sigma -- standard deviation \\ 
            mu -- mean input \\
            omega -- signal frequency\n

        Keyword arguments of MCF method: \\
            l_max -- number of correction terms included in the pertubation series \\
            n_max,p_max -- number of Fourier modes, Hermite functions included \\
    '''

    r = np.zeros((l_max+1,l_max+1),dtype=np.complex_)

    c_p = np.zeros((l_max+1,n_max+1,p_max),dtype=np.complex_)
    c_n = np.zeros((l_max+1,n_max+1,p_max),dtype=np.complex_)

    c_p_previous = np.zeros((l_max+1,n_max+1,p_max),dtype=np.complex_)
    c_n_previous = np.zeros((l_max+1,n_max+1,p_max),dtype=np.complex_)


    for l in range(l_max+1):

        for k in range(l_max+1):

            # The upcoming condition reflects that most of the responsefunctions vanish.
            # Thus, we only have to perform the MCF method for the non-vanishing terms.
            if k > l or (k-l) % 2 == 1:

                c_p[k,:,:] = np.zeros((n_max+1,p_max),dtype=np.complex_)
                c_n[k,:,:] = np.zeros((n_max+1,p_max),dtype=np.complex_)

            else:

                if k == l:
                    c_p_prevsum = np.copy(c_p_previous[k-1,:,:])
                    c_n_prevsum = np.copy(c_n_previous[k-1,:,:])      
                else:
                    if k == 0:
                        c_p_prevsum = np.copy(c_p_previous[k+1,:,:]+np.conj(c_n_previous[k+1,:,:]))
                        c_n_prevsum = np.copy(c_n_previous[k+1,:,:]+np.conj(c_p_previous[k+1,:,:]))
                    else:    
                        c_p_prevsum = np.copy(c_p_previous[k-1,:,:]+c_p_previous[k+1,:,:])
                        c_n_prevsum = np.copy(c_n_previous[k-1,:,:]+c_n_previous[k+1,:,:])
                    
                # compute responsfunction r[k,l] and coefficients c_out
                r[l,k],c_p[k,:n_max-l,:],c_n[k,:n_max-l,:] = single_response_func_cosine_signal(l,k,n_max-l-1,p_max,tau,sigma,mu,omega,c_p_prevsum,c_n_prevsum)

        c_p_previous = np.copy(c_p)
        c_n_previous = np.copy(c_n)

        print('finished computing response funcs with order l =',l)

    return r



def response_cosine_signal(epsilon,omega,response_funcs,t_start,t_end,t_res):
    '''
        Returns the cyclo-stationary response r(t) 
        to a periodic stimulus s(t) = epsilon*cos(omega*t) 
        in the time window [t_start,t_end] with a resolution t_res\n

        The underlying model is a theta neuron which is driven by
        correlated noise and the signal s(t) \n

        The model parameter namely mean input, variance of noise
        and correlation time are already encoded in the 
        response functions: response_funcs\n

        Keyword arguments:\\
            epsilon -- amplitude of stimulus\\
            omega -- frequency of stimulus\\
            response_funcs -- response functions \\
            t_start,t_end,t_res -- time interval
    '''

    t = np.linspace(t_start,t_end,t_res)
    l_max,_ = response_funcs.shape 
    r = np.zeros(len(t))

    amp_mod = np.absolute(response_funcs)    # amplitude modulation factors
    phase_shift = np.angle(response_funcs)   # phase shifts

    for l in range(l_max):
        for k in range(l_max):
            r += pow(epsilon,l)*amp_mod[l,k]*np.cos(k*omega*t-phase_shift[l,k])

    return r,t




##########################################
########## TWO PERIODIC STIMULI ##########
##########################################


def response_funcs_two_cosine_signals(tau,sigma,mu,omega_1,omega_2,n_max=100,p_max=100):
    '''
        Computes ALL the response function up to the 2nd order
        of the theta neuron driven by correlated noise and two cosine stimuli
        with matrix continued fraction (MCF) method \n

        Keyword arguments - stochastic system: \\
            tau -- correlation time \\
            sigma -- standard deviation \\ 
            mu -- mean input \\
            omega_1 -- frequency of stimulus 1 \\
            omega_2 -- frequency of stimulus 2 \n

        Keyword arguments - MCF method: \\
            n_max,p_max -- number of Fourier modes, Hermite functions included \\
    '''

    l_max = 2 # up to 2nd order

    r_1 = np.zeros((l_max+1,l_max+1),dtype=np.complex_)
    r_2 = np.zeros((l_max+1,l_max+1),dtype=np.complex_)

    c_p_1 = np.zeros((l_max+1,n_max+1,p_max),dtype=np.complex_)
    c_n_1 = np.zeros((l_max+1,n_max+1,p_max),dtype=np.complex_)
    c_p_2 = np.zeros((l_max+1,n_max+1,p_max),dtype=np.complex_)
    c_n_2 = np.zeros((l_max+1,n_max+1,p_max),dtype=np.complex_)

    c_p_1_prev = np.zeros((l_max+1,n_max+1,p_max),dtype=np.complex_)
    c_n_1_prev = np.zeros((l_max+1,n_max+1,p_max),dtype=np.complex_)

    # compute stationary firing rate
    # r_1[0,0] corresponds to r_{0,0}
    l = 0
    r_1[0,0],c_p_1[0,:n_max-l,:],c_n_1[0,:n_max-l,:] = single_response_func_cosine_signal(0,0,n_max-l-1,p_max,tau,sigma,mu,0,c_p_1_prev[0,:,:],c_n_1_prev[0,:,:])

    r_2[0,0] = r_1[0,0]
    c_p_2 = np.copy(c_p_1)
    c_n_2 = np.copy(c_n_1)


    # compute the rate modulation for stimulus 1 and stimulus 2 (uncoupled)  
    # r_1[l,k] corresponds to r_{l,k}(omega_1) in README or r_{l,0}^{k,0} in paper
    for l in range(1,l_max+1):

        c_p_1_prev = np.copy(c_p_1)
        c_n_1_prev = np.copy(c_n_1)
        c_p_2_prev = np.copy(c_p_2)
        c_n_2_prev = np.copy(c_n_2)

        c_p_1 = np.zeros((l_max+1,n_max+1,p_max),dtype=np.complex_)
        c_n_1 = np.zeros((l_max+1,n_max+1,p_max),dtype=np.complex_)
        c_p_2 = np.zeros((l_max+1,n_max+1,p_max),dtype=np.complex_)
        c_n_2 = np.zeros((l_max+1,n_max+1,p_max),dtype=np.complex_)

        for k in range(l_max+1):
            
            if k > l or (k-l) % 2 == 1:

                r_1[l,k] = 0
                r_2[l,k] = 0

            else:

                if k == l:
                    c_p_1_prevsum = c_p_1_prev[k-1,:,:]
                    c_n_1_prevsum = c_n_1_prev[k-1,:,:]
                    c_p_2_prevsum = c_p_2_prev[k-1,:,:]
                    c_n_2_prevsum = c_n_2_prev[k-1,:,:]
                else:
                    if k == 0:
                        c_p_1_prevsum = np.copy(c_p_1_prev[k+1,:,:]+np.conj(c_n_1_prev[k+1,:,:]))
                        c_n_1_prevsum = np.copy(c_n_1_prev[k+1,:,:]+np.conj(c_p_1_prev[k+1,:,:]))
                        c_p_2_prevsum = np.copy(c_p_2_prev[k+1,:,:]+np.conj(c_n_2_prev[k+1,:,:]))
                        c_n_2_prevsum = np.copy(c_n_2_prev[k+1,:,:]+np.conj(c_p_2_prev[k+1,:,:]))
                    else:    
                        c_p_1_prevsum = (c_p_1_prev[k-1,:,:]+c_p_1_prev[k+1,:,:])
                        c_n_1_prevsum = (c_n_1_prev[k-1,:,:]+c_n_1_prev[k+1,:,:])
                        c_p_2_prevsum = (c_p_2_prev[k-1,:,:]+c_p_2_prev[k+1,:,:])
                        c_n_2_prevsum = (c_n_2_prev[k-1,:,:]+c_n_2_prev[k+1,:,:])
                    
                r_1[l,k],c_p_1[k,:n_max-l,:],c_n_1[k,:n_max-l,:] = single_response_func_cosine_signal(l,k,n_max-l-1,p_max,tau,sigma,mu,omega_1,c_p_1_prevsum,c_n_1_prevsum)
                r_2[l,k],c_p_2[k,:n_max-l,:],c_n_2[k,:n_max-l,:] = single_response_func_cosine_signal(l,k,n_max-l-1,p_max,tau,sigma,mu,omega_2,c_p_2_prevsum,c_n_2_prevsum)


    # compute response functions which take into account that both signals enter neuron simultaniously
    # r_mix_sum corresponds to r_mix(omega_1,omega_2) in README or r_{1,1}^{1,1} in paper
    # r_mix_diff corresponds to r_mix(omega_1,-omega_2) in README or r_{1,1}^{1,-1} in paper
    l = 1
    omega = omega_1 + omega_2
    
    r_mix_sum,_,_ = single_response_func_cosine_signal(1,1,n_max-l-1,p_max,tau,sigma,mu,omega,(c_p_1_prev[1,:,:]+c_p_2_prev[1,:,:]),(c_n_1_prev[1,:,:]+c_n_2_prev[1,:,:]))

    omega = omega_1 - omega_2
    r_mix_diff,_,_ = single_response_func_cosine_signal(1,1,n_max-l-1,p_max,tau,sigma,mu,omega,(c_p_1_prev[1,:,:]+np.conj(c_n_2_prev[1,:,:])),(c_n_1_prev[1,:,:]+np.conj(c_p_2_prev[1,:,:])))

    
    return (r_1,r_2,r_mix_sum,r_mix_diff)




def response_two_cosine_signals(epsilon_1,omega_1,epsilon_2,omega_2,response_func,t_start,t_end,t_res):
    '''
        This function returns the response to two periodic stimuli
        s(t) = epsilon_1*cos(omega_1*t) + epsilon_2*cos(omega_2*t) 
        in the time window [t_start,t_end] with a resolution t_res\n

        The underlying model is a theta neuron which is driven by
        correlated noise and the signal s(t) \n

        The model parameter namely mean input, variance of noise
        and correlation time are already encoded in the response functions: 
        response_func = [r_func_1,r_func_2,r_func_add,r_func_diff]\n

        r_func_1 ... response functions for single signal 1 \\
        r_func_2 ... response functions for single signal 2 \\
        r_func_sum, r_func_diff ... response functions representing 
        that both signals enter neuron at the same time
    '''

    r_func_1 = response_func[0]
    r_func_2 = response_func[1]
    r_func_sum = response_func[2]
    r_func_diff = response_func[3]

    # set the stationary / unperturbed firing rate of individual response 2 to zero 
    # so that we dont add the stationary rate twice
    r_func_2[0,0] = 0            


    ### 1: sum of responses to individual signals
    response_1,_ = response_cosine_signal(epsilon_1,omega_1,r_func_1,t_start,t_end,t_res)
    response_2,t = response_cosine_signal(epsilon_2,omega_2,r_func_2,t_start,t_end,t_res)

    response = response_1 + response_2


    ### 2: add correction, which represents that both signals enter neuron simultaneously
    response += epsilon_1*epsilon_2*np.absolute(r_func_sum)*np.cos((omega_1+omega_2)*t-np.angle(r_func_sum))
    response += epsilon_1*epsilon_2*np.absolute(r_func_diff)*np.cos((omega_1-omega_2)*t-np.angle(r_func_diff))


    return response,t