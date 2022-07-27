import MCF_torch
import custom_plots
import numpy as np
import pickle
import matplotlib.pyplot as plt


### import the prerecorded data and corresponding parameters
f = open('data/simulation_stationary_firing_3' + '.pckl', 'rb')
tau_sim,firing_rate_sim,sigma,mu = pickle.load(f)
f.close()


### MCF method: compute the firing rate for multiple correlation times tau
res = 50
tau = np.logspace(np.log10(0.02),np.log10(10),res)
n_max = [8,16,32,64,128]
firing_rate = np.zeros((len(n_max),res))

for j in range(len(n_max)):
    for i in range(res): 
        firing_rate[j,i] = MCF_torch.stationary_firing_rate(tau[i],sigma,mu,n_max = n_max[j],p_max = n_max[j])
        print('finished computing datapoint',i+1,'/',res)



### plot the firing rate
title='Mean-driven Regime'

f0 = 10

fig = plt.figure(figsize=(4.5,4))
ax = fig.add_subplot()
fig.suptitle(title.upper(), fontsize=12)
ax.set_title(r'$\mu = $'+str(mu)+'; $\sigma=$'+str(sigma), fontsize=f0-1.5, pad=8)

ax.scatter(tau_sim,firing_rate_sim,s=8,label='Simulation',zorder=2)
for j in range(len(n_max)-1,-1,-1):
    ax.plot(tau,firing_rate[j],label=r'$n_{max} = p_{max} = $' + str(n_max[j]),zorder=3+i,lw = 3.5/((len(n_max)-j)))

    if j == 0:
        y_min,y_max = ax.get_ylim()

plt.xscale('log')
plt.xlim(tau[0],tau[-1])
ax.set_ylim(max(y_min,-0.005),y_max)

ax.set_xlabel(r'Correlation time',fontsize=f0)
ax.set_ylabel(r'Stat. firing rate',fontsize=f0)
ax.tick_params(labelsize=f0-3)
ax.tick_params(labelsize=f0-3)

if mu > 1:
    loc_legend = 1
else:
    loc_legend = 0

#ax.legend(loc=loc_legend,fancybox=True,fontsize=f0-1.5)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.grid(True,ls='dotted')

plt.show()