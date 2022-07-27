import matplotlib.pyplot as plt

def plot_response(t,signal,lines_resp_y,lines_resp_x,lines_resp_label,parameters,title):

    f0 = 10
    f1 = 9
    f2 = 7.5

    fig = plt.figure(figsize=(8,4))

    grid = fig.add_gridspec(3, 3)

    ax1 = fig.add_subplot(grid[0,:-1])
    ax2 = fig.add_subplot(grid[1:,:-1])

    ax1.set_title(title.upper(), fontsize=12, pad=10)

    ax1.plot(t,signal,color='tab:gray')

    colors = ('cornflowerblue','k','tab:red')
    
    for i in range(len(lines_resp_x)):
        ax2.plot(lines_resp_x[i],lines_resp_y[i],label=lines_resp_label[i],color=colors[i])

    ax1.set_xlim(t[0],t[-1])
    ax2.set_xlim(t[0],t[-1])

    # only for not trimmed simulation data
    ax2.legend(loc=1, fancybox=True,fontsize=f2)
    ax1.set_ylabel(r'Signal',fontsize=f0)
    ax2.set_ylabel(r'Firing rate',fontsize=f0)
    ax2.set_xlabel(r'Time',fontsize=f0)
    ax1.tick_params(labelbottom=False,labelsize=f0-3)
    ax2.tick_params(labelsize=f0-3)

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    ax1.grid(True,ls='dotted')
    ax2.grid(True,ls='dotted')

    ### Print the parameters next to the plot
    ax3 = fig.add_subplot(grid[:,-1])

    h_noise = 0.88
    spacing = 0.06
    h_signal = h_noise - 5*spacing
    tab = .06

    ax3.text(0,1,'Parameters',va='top',fontweight='semibold',fontsize=f1)
    ax3.text(0,h_noise+.1*spacing,'Noise:',va='top',fontsize=f1)
    ax3.text(tab,h_noise-spacing,r'Mean $\mu = $'+str(parameters[0]),va='top',fontsize=f2)
    ax3.text(tab,h_noise-2*spacing,r'Variance $\sigma^2 = $'+str(pow(parameters[1],2)),va='top',fontsize=f2)
    ax3.text(tab,h_noise-3*spacing,r'Correlation time $\tau = $'+str(parameters[2]),va='top',fontsize=f2)

    ax3.text(0,h_signal+.1*spacing,'Signal:',va='top',fontsize=f1)

    # showing parameters for two signals
    if len(parameters) == 7:
        ax3.text(tab,h_signal-spacing,r'Amplitudes $\varepsilon_1 = $'+str(parameters[3])+r', $\varepsilon_2 = $'+str(parameters[5]),va='top',fontsize=f2)
        ax3.text(tab,h_signal-2*spacing,r'Frequencies $\omega_1 = $'+str(parameters[4])+r', $\omega_2 = $'+str(parameters[6]),va='top',fontsize=f2)

    # showing parameters for one signal
    if len(parameters) == 5:
        ax3.text(tab,h_signal-spacing,r'Amplitude $\varepsilon = $'+str(parameters[3]),va='top',fontsize=f2)
        ax3.text(tab,h_signal-2*spacing,r'Frequency $\omega = $'+str(parameters[4]),va='top',fontsize=f2)

    plt.axis('off')

    plt.show()




def plot_stationary_rate(tau,firing_rate,tau_sim,firing_rate_sim,parameters,title):

    f0 = 10

    mu = parameters[0]
    sigma = parameters[1]

    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot()
    fig.suptitle(title.upper(), fontsize=12)
    ax.set_title(r'for $\mu = $'+str(mu)+'; $\sigma=$'+str(sigma), fontsize=f0-1.5, pad=8)

    ax.scatter(tau_sim,firing_rate_sim,s=8,label='Simulation',zorder=2)
    ax.plot(tau,firing_rate,color='k',label='MCF method',zorder=3)

    plt.xscale('log')
    plt.xlim(tau[0],tau[-1])

    ax.set_xlabel(r'Correlation time',fontsize=f0)
    ax.set_ylabel(r'Firing rate',fontsize=f0)
    ax.tick_params(labelsize=f0-3)
    ax.tick_params(labelsize=f0-3)
    
    if mu > 1:
        loc_legend = 1
    else:
        loc_legend = 0
    
    ax.legend(loc=loc_legend,fancybox=True,fontsize=f0-1.5)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(True,ls='dotted')

    plt.show()



def plot_stat_simulation(tau_sim,firing_rate_sim,parameters,title):

    f0 = 10

    mu = parameters[0]
    sigma = parameters[1]

    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot()
    fig.suptitle(title.upper(), fontsize=12)
    ax.set_title(r'for $\mu = $'+str(mu)+'; $\sigma=$'+str(sigma), fontsize=f0-1.5, pad=8)

    ax.scatter(tau_sim,firing_rate_sim,s=8,label='Simulation',zorder=2)

    plt.xscale('log')

    ax.set_xlabel(r'Correlation time',fontsize=f0)
    ax.set_ylabel(r'Firing rate',fontsize=f0)
    ax.tick_params(labelsize=f0-3)
    ax.tick_params(labelsize=f0-3)
    
    if mu > 1:
        loc_legend = 1
    else:
        loc_legend = 0
    
    ax.legend(loc=loc_legend,fancybox=True,fontsize=f0-1.5)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(True,ls='dotted')

    plt.show()