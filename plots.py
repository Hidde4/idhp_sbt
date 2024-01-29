import matplotlib.pyplot as plt
import numpy as np
import os

def plot_baseline(task, path):

    state_history = np.load(os.path.join(path, f"state_history.npy"))
    action_history = np.load(os.path.join(path, f"action_history.npy"))

    fig, ax = plt.subplots(5)
    fig.tight_layout()
    fig.set_figwidth(4.15)
    fig.set_figheight(5.5)
    ax[0].plot(task.t_range,action_history*180/np.pi,'b')
    ax[0].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[0].set_ylabel(r"$\gamma$ [deg]")
    ax[1].plot(task.t_range,state_history[0,:]*180/np.pi,'b')
    ax[1].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[1].set_ylabel(r"$v$ [m/s]")
    ax[2].plot(task.t_range,state_history[1,:]*180/np.pi,'b',label=r'$\phi$')
    ax[2].plot(task.t_range,task.bank_reference*180/np.pi,'r--',label=r'$\phi_{ref}$')
    ax[2].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[2].set_ylabel(r"$\phi$ [deg]")
    ax[2].legend(fontsize='6',loc='upper right')
    ax[3].plot(task.t_range,state_history[2,:]*180/np.pi,'b')
    ax[3].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[3].set_ylabel(r"$p$ [deg/s]")
    ax[4].plot(task.t_range,state_history[3,:]*180/np.pi,'b')
    ax[4].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[4].set_xlabel(r"$Time$ [s]")
    ax[4].set_ylabel(r"$r$ [deg/s]")
    fig.align_ylabels()
    plt.show()
    
    fig.savefig(os.path.join(path, "state_tracking.pdf"))

def plot_random_init(task, path):
    all_state_history  = []
    all_action_history = []
    for i in range(20):
        all_state_history.append(np.load(os.path.join(path, f"state_history_{i}.npy")))
        all_action_history.append(np.load(os.path.join(path, f"action_history_{i}.npy")))

    state_history1_min = np.min([state[0] for state in all_state_history],axis=0)
    state_history1_max = np.max([state[0] for state in all_state_history],axis=0)
    state_history1_mean = np.mean([state[0] for state in all_state_history],axis=0)

    state_history2_min = np.min([state[1] for state in all_state_history],axis=0)
    state_history2_max = np.max([state[1] for state in all_state_history],axis=0)
    state_history2_mean = np.mean([state[1] for state in all_state_history],axis=0)

    state_history3_min = np.min([state[2] for state in all_state_history],axis=0)
    state_history3_max = np.max([state[2] for state in all_state_history],axis=0)
    state_history3_mean = np.mean([state[2] for state in all_state_history],axis=0)

    state_history4_min = np.min([state[3] for state in all_state_history],axis=0)
    state_history4_max = np.max([state[3] for state in all_state_history],axis=0)
    state_history4_mean = np.mean([state[3] for state in all_state_history],axis=0)

    action_history_min = np.min(all_action_history,axis=0)
    action_history_max = np.max(all_action_history,axis=0)
    action_history_mean = np.mean(all_action_history,axis=0)

    fig, ax = plt.subplots(5)
    fig.tight_layout()
    fig.set_figwidth(4.15)
    fig.set_figheight(5.5)
    ax[0].plot(task.t_range,action_history_mean*180/np.pi,'b')
    ax[0].fill_between(task.t_range,action_history_min*180/np.pi,action_history_max*180/np.pi,color='b',alpha=0.4)
    ax[0].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[0].set_ylabel(r"$\gamma$ [deg]")
    ax[1].plot(task.t_range,state_history1_mean*180/np.pi,'b')
    ax[1].fill_between(task.t_range,state_history1_min*180/np.pi,state_history1_max*180/np.pi,color='b',alpha=0.4)
    ax[1].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[1].set_ylabel(r"$v$ [m/s]")
    ax[2].plot(task.t_range,state_history2_mean*180/np.pi,'b',label=r'$\phi$')
    ax[2].fill_between(task.t_range,state_history2_min*180/np.pi,state_history2_max*180/np.pi,color='b',alpha=0.4)
    ax[2].plot(task.t_range,task.bank_reference*180/np.pi,'r--',label=r'$\phi_{ref}$')
    ax[2].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[2].set_ylabel(r"$\phi$ [deg]")
    ax[2].legend(fontsize='6',loc='upper right')
    ax[3].plot(task.t_range,state_history3_mean*180/np.pi,'b')
    ax[3].fill_between(task.t_range,state_history3_min*180/np.pi,state_history3_max*180/np.pi,color='b',alpha=0.4)
    ax[3].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[3].set_ylabel(r"$p$ [deg/s]")
    ax[4].plot(task.t_range,state_history4_mean*180/np.pi,'b')
    ax[4].fill_between(task.t_range,state_history4_min*180/np.pi,state_history4_max*180/np.pi,color='b',alpha=0.4)
    ax[4].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[4].set_xlabel(r"$Time$ [s]")
    ax[4].set_ylabel(r"$r$ [deg/s]")
    fig.align_ylabels()
    plt.show()
    
    fig.savefig(os.path.join(path, "state_tracking_init.pdf"))


def plot_reward_scale(task, path):

    fig, ax = plt.subplots(5)
    fig.tight_layout()
    fig.set_figwidth(4.15)
    fig.set_figheight(5.5)
    for i in range(10,110,20):
        state_history = (np.load(os.path.join(path, f"state_history_{i}.npy")))
        action_history = (np.load(os.path.join(path, f"action_history_{i}.npy")))
        ax[0].plot(task.t_range,action_history*180/np.pi)
        ax[1].plot(task.t_range,state_history[0,:]*180/np.pi)
        ax[2].plot(task.t_range,state_history[1,:]*180/np.pi,label=f"$\kappa = $ {i}")
        ax[3].plot(task.t_range,state_history[2,:]*180/np.pi)
        ax[4].plot(task.t_range,state_history[3,:]*180/np.pi)
    
    ax[0].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[0].set_ylabel(r"$\gamma$ [deg]")
    ax[1].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[1].set_ylabel(r"$v$ [m/s]")
    ax[2].plot(task.t_range,task.bank_reference*180/np.pi,'r--',label=r'$\phi_{ref}$')
    ax[2].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[2].set_ylabel(r"$\phi$ [deg]")
    ax[3].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[3].set_ylabel(r"$p$ [deg/s]")
    ax[4].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[4].set_xlabel(r"$Time$ [s]")
    ax[4].set_ylabel(r"$r$ [deg/s]")
    fig.legend(fontsize='6',loc='center',bbox_to_anchor=(0.6, 0.6), ncol=3)
    fig.align_ylabels()
    plt.show()
    
    fig.savefig(os.path.join(path, "state_tracking_reward.pdf"))

def plot_neurons(task, path):

    fig, ax = plt.subplots(5)
    # fig.tight_layout()
    fig.set_figwidth(4.15)
    fig.set_figheight(5.5)
    for i in range(10,60,10):
        state_history = (np.load(os.path.join(path, f"state_history_{i}.npy")))
        action_history = (np.load(os.path.join(path, f"action_history_{i}.npy")))
        ax[0].plot(task.t_range,action_history*180/np.pi)
        ax[1].plot(task.t_range,state_history[0,:]*180/np.pi)
        ax[2].plot(task.t_range,state_history[1,:]*180/np.pi,label=f"$n = $ {i}")
        ax[3].plot(task.t_range,state_history[2,:]*180/np.pi)
        ax[4].plot(task.t_range,state_history[3,:]*180/np.pi)
    
    
    ax[0].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[0].set_ylabel(r"$\gamma$ [deg]")
    ax[1].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[1].set_ylabel(r"$v$ [m/s]")
    ax[2].plot(task.t_range,task.bank_reference*180/np.pi,'r--',label=r'$\phi_{ref}$')
    ax[2].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[2].set_ylabel(r"$\phi$ [deg]") 
    ax[3].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[3].set_ylabel(r"$p$ [deg/s]")
    ax[4].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[4].set_xlabel(r"$Time$ [s]")
    ax[4].set_ylabel(r"$r$ [deg/s]")
    fig.legend(fontsize='6',loc='center',bbox_to_anchor=(0.6, 0.6), ncol=3)
    fig.align_ylabels()
    plt.show()
    
    fig.savefig(os.path.join(path, "state_tracking_neurons.pdf"))

def plot_cg(task, path):

    state_history = np.load(os.path.join(path, f"state_history.npy"))
    action_history = np.load(os.path.join(path, f"action_history.npy"))

    fig, ax = plt.subplots(5)
    fig.tight_layout()
    fig.set_figwidth(4.15)
    fig.set_figheight(5.5)
    ax[0].plot(task.t_range,action_history*180/np.pi,'b')
    ax[0].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[0].set_ylabel(r"$\gamma$ [deg]")
    ax[1].plot(task.t_range,state_history[0,:]*180/np.pi,'b')
    ax[1].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[1].set_ylabel(r"$v$ [m/s]")
    ax[2].plot(task.t_range,state_history[1,:]*180/np.pi,'b',label=r'$\phi$')
    ax[2].plot(task.t_range,task.bank_reference*180/np.pi,'r--',label=r'$\phi_{ref}$')
    ax[2].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[2].set_ylabel(r"$\phi$ [deg]")
    ax[2].legend(fontsize='6',loc='upper right')
    ax[3].plot(task.t_range,state_history[2,:]*180/np.pi,'b')
    ax[3].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[3].set_ylabel(r"$p$ [deg/s]")
    ax[4].plot(task.t_range,state_history[3,:]*180/np.pi,'b')
    ax[4].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[4].set_xlabel(r"$Time$ [s]")
    ax[4].set_ylabel(r"$r$ [deg/s]")
    fig.align_ylabels()
    plt.show()
    
    fig.savefig(os.path.join(path, "state_tracking_cg.pdf"))

def plot_reduced_steering(task, path):

    state_history = np.load(os.path.join(path, f"state_history.npy"))
    action_history = np.load(os.path.join(path, f"action_history.npy"))

    fig, ax = plt.subplots(5)
    fig.tight_layout()
    fig.set_figwidth(4.15)
    fig.set_figheight(5.5)
    ax[0].plot(task.t_range,action_history*180/np.pi,'b')
    ax[0].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[0].set_ylabel(r"$\gamma$ [deg]")
    ax[1].plot(task.t_range,state_history[0,:]*180/np.pi,'b')
    ax[1].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[1].set_ylabel(r"$v$ [m/s]")
    ax[2].plot(task.t_range,state_history[1,:]*180/np.pi,'b',label=r'$\phi$')
    ax[2].plot(task.t_range,task.bank_reference*180/np.pi,'r--',label=r'$\phi_{ref}$')
    ax[2].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[2].set_ylabel(r"$\phi$ [deg]")
    ax[2].legend(fontsize='6',loc='upper right')
    ax[3].plot(task.t_range,state_history[2,:]*180/np.pi,'b')
    ax[3].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[3].set_ylabel(r"$p$ [deg/s]")
    ax[4].plot(task.t_range,state_history[3,:]*180/np.pi,'b')
    ax[4].set_xlim([task.t_range[0],task.t_range[-1]+0.01])
    ax[4].set_xlabel(r"$Time$ [s]")
    ax[4].set_ylabel(r"$r$ [deg/s]")
    fig.align_ylabels()
    plt.show()
    
    fig.savefig(os.path.join(path, "state_tracking_reduced_steering.pdf"))

def plot_weights(agent, task, path):

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[1, 1, 1, 1])
    fig.set_figwidth(4.15)
    fig.set_figheight(5.5)
    
    actor_weights_l1 = np.array([_[-2].flatten() for _ in agent.actor_weights_history])
    actor_weights_l2 = np.array([_[-1].flatten() for _ in agent.actor_weights_history])

    critic_weights_l1 = np.array([_[-2].flatten() for _ in agent.critic_weights_history])
    critic_weights_l2 = np.array([_[-1].flatten() for _ in agent.critic_weights_history])

    # Plot: actor weights: layer 1
    ax = fig.add_subplot(gs[0])
    ax.set_xlabel(r"$Time$ [s]", labelpad=10)
    ax.set_ylabel(r"$\theta$ [-]", labelpad=10)

    for i in range(actor_weights_l1.shape[1]):
        ax.plot(task.t_range, actor_weights_l1[:, i])

    # Plot: critic weights
    ax = fig.add_subplot(gs[2])
    ax.set_xlabel(r"$Time$ [s]", labelpad=10)
    ax.set_ylabel(r"$w$ [-]", labelpad=10)

    for i in range(critic_weights_l1.shape[1]):
        ax.plot(task.t_range, critic_weights_l1[:, i])

    # Plot: actor weights: layer 1
    ax = fig.add_subplot(gs[1])
    ax.set_xlabel(r"$Time$ [s]", labelpad=10)
    ax.set_ylabel(r"$\theta$ [-]", labelpad=10)

    for i in range(actor_weights_l2.shape[1]):
        ax.plot(task.t_range, actor_weights_l2[:, i])

    # Plot: critic weights
    ax = fig.add_subplot(gs[3])
    ax.set_xlabel(r"$Time$ [s]", labelpad=10)
    ax.set_ylabel(r"$w$ [-]", labelpad=10)

    for i in range(critic_weights_l2.shape[1]):
        ax.plot(task.t_range, critic_weights_l2[:, i])
    fig.align_ylabels()
    plt.show()
    fig.savefig(os.path.join(path, "weights.pdf"))