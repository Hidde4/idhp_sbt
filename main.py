from task import BankAngleTracking
from incremental_model import IncrementalModel
from environment import SolarBoat
import matplotlib.pyplot as plt
import numpy as np
from agent import AgentIDHP
from plots import *
import os
import matplotlib as mpl

# Plot parameters, adapted from Teirlinck
mpl.rcParams["figure.autolayout"] = True
mpl.rcParams["lines.linewidth"] = 1.0
mpl.rcParams["grid.color"] = "C1C1C1"
mpl.rcParams["grid.linestyle"] = ":"
mpl.rcParams["axes.grid"] = True
mpl.rcParams["axes.xmargin"] = 0
mpl.rcParams["axes.ymargin"] = 0.1
mpl.rcParams["axes.labelpad"] = 4.0
mpl.rcParams["legend.framealpha"] = 1.0
mpl.rcParams["legend.edgecolor"] = "k"
mpl.rcParams["legend.fancybox"] = False
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["ytick.minor.visible"] = True
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["savefig.bbox"] = "tight"

def main():
    task = BankAngleTracking(T=20,dt=0.01)

    # Baseline -----------------------
    path = f"results/baseline"
    if not os.path.exists(path):
        os.makedirs(path)
    agent = train_baseline(task, path)
    plot_weights(agent, task, path)
    plot_baseline(task,path)

    # Random intialization ------------
    # path = f"results/random_init"
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # # train_random_init(task, path)

    # plot_random_init(task,path)

    # Varying reward scale -------------
    # path = f"results/reward_scale"
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # train_reward_scale(task, path)

    # plot_reward_scale(task,path)

    # Varying neurons ------------------
    # path = f"results/neurons"
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # train_neurons(task, path)

    # plot_neurons(task,path)

    # Cg shift -------------------------
    # path = f"results/cg_shift"
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # train_cg(task, path)

    # plot_cg(task,path)

     # Reduced steering angle effectiveness 75%
    # path = f"results/reduced_steering"
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # train_reduced_steering(task, path)

    # plot_reduced_steering(task,path)

def train_baseline(task, path):
    
    env = SolarBoat(task)

    PARAMETERS = {
                "gamma_model": 1.0,
                "cov_init": 1.0e8,
                "neurons": 20,
                "std_init": 0.05,
                "seed":10,
                "lr":1e-3,
                "gamma_discount":0.9,
                "reward_scale":50.0,
                "trim":True,
                "cg_shift":False,
                "steering_fail":False}
    
    agent = AgentIDHP(env, task, PARAMETERS)

    agent.learn()
        
    np.save(os.path.join(path,f"state_history"),env.state_history)
    np.save(os.path.join(path,f"action_history"),env.action_history)

    return agent

def train_cg(task, path):
    
    env = SolarBoat(task)

    PARAMETERS = {
                "gamma_model": 1.0,
                "cov_init": 1.0e8,
                "neurons": 20,
                "std_init": 0.05,
                "seed":10,
                "lr":1e-3,
                "gamma_discount":0.9,
                "reward_scale":50.0,
                "trim":True,
                "cg_shift":True,
                "steering_fail":False}
    
    agent = AgentIDHP(env, task, PARAMETERS)

    agent.learn()
        
    np.save(os.path.join(path,f"state_history"),env.state_history)
    np.save(os.path.join(path,f"action_history"),env.action_history)

def train_reduced_steering(task, path):
    
    env = SolarBoat(task)

    PARAMETERS = {
                "gamma_model": 1.0,
                "cov_init": 1.0e8,
                "neurons": 20,
                "std_init": 0.05,
                "seed":10,
                "lr":1e-3,
                "gamma_discount":0.9,
                "reward_scale":50.0,
                "trim":True,
                "cg_shift":False,
                "steering_fail":True}
    
    agent = AgentIDHP(env, task, PARAMETERS)

    agent.learn()
        
    np.save(os.path.join(path,f"state_history"),env.state_history)
    np.save(os.path.join(path,f"action_history"),env.action_history)
 
def train_random_init(task, path):
    # Train for different random initializations of the parameters

    PARAMETERS = {
                "gamma_model": 1.0,
                "cov_init": 1.0e8,
                "neurons": 20,
                "std_init": 0.05,
                "seed":None,
                "lr":1e-3,
                "gamma_discount":0.9,
                "reward_scale":50.0,
                "trim":False,
                "cg_shift":False,
                "steering_fail":False}
    
    for i in range(20):
        print(f'Iteration = {i+1}')
        env = SolarBoat(task)
        agent = AgentIDHP(env, task, PARAMETERS)

        agent.learn()
            
        np.save(os.path.join(path,f"state_history_{i}"),env.state_history)
        np.save(os.path.join(path,f"action_history_{i}"),env.action_history)

def train_reward_scale(task, path):
    # Train for different random initializations of the parameters

    PARAMETERS = {
                "gamma_model": 1.0,
                "cov_init": 1.0e8,
                "neurons": 20,
                "std_init": 0.05,
                "seed":10,
                "lr":1e-3,
                "gamma_discount":0.9,
                "reward_scale":50.0,
                "trim":True,
                "cg_shift":False,
                "steering_fail":False}
    it = 1
    for i in range(10,110,20):
        PARAMETERS["reward_scale"] = i
        print(f'Iteration = {it}')
        env = SolarBoat(task)
        agent = AgentIDHP(env, task, PARAMETERS)

        agent.learn()
            
        np.save(os.path.join(path,f"state_history_{i}"),env.state_history)
        np.save(os.path.join(path,f"action_history_{i}"),env.action_history)
        it += 1

def train_neurons(task, path):
    # Train for different random initializations of the parameters

    PARAMETERS = {
                "gamma_model": 1.0,
                "cov_init": 1.0e8,
                "neurons": 20,
                "std_init": 0.05,
                "seed":10,
                "lr":1e-3,
                "gamma_discount":0.9,
                "reward_scale":50.0,
                "trim":True,
                "cg_shift":False,
                "steering_fail":False}
    it = 1
    for i in range(10,60,10):
        PARAMETERS["neurons"] = i
        print(f'Iteration = {it}')
        env = SolarBoat(task)
        agent = AgentIDHP(env, task, PARAMETERS)

        agent.learn()
            
        np.save(os.path.join(path,f"state_history_{i}"),env.state_history)
        np.save(os.path.join(path,f"action_history_{i}"),env.action_history)
        it += 1

if __name__ == "__main__":
    main()