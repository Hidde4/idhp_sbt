import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from actor import Actor
from critic import Critic
from incremental_model import IncrementalModel
import random
from tqdm import tqdm

class AgentIDHP:

    def __init__(self, env, task,  parameters):
        
        seed = (parameters["seed"])
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)

         # Environment
        self.env = env
        self.trim = parameters["trim"]
        self.cg = parameters["cg_shift"]
        self.steering_fail = parameters["steering_fail"]

        # Task
        self.task = task
        self.num_timesteps = len(self.task.t_range)

        # Actor
        self.actor = Actor(self.env, parameters)

        # Critic
        self.critic = Critic(self.env, parameters)
        # self.critic_target = Critic(self.env, parameters)

         # Incremental model
        self.incremental_model = IncrementalModel(self.env, parameters)

        # Training
        self.lr = parameters["lr"]
        self.gamma_discount = parameters["gamma_discount"]
        
        self.reward_scale = parameters["reward_scale"]
        self.t = 0
        self.tau = 0.01
        # Optimizers
        self.actor_optimizer = SGD(tf.Variable(self.lr, trainable=False))
        self.critic_optimizer = SGD(tf.Variable(self.lr, trainable=False))

        # Logging
        self.actor_weights_history = []
        self.critic_weights_history = []
        self.F_history = []
        self.G_history = []
        self.cov_history = []
        self.epsilon_history = []
        

    @tf.function(experimental_relax_shapes=True)
    def call(self, obs):
        
        action = self.actor(obs)
        action = tf.reshape(action, [-1])  # remove batch dimension

        return action
    
    def learn(self):

        # Initialize environment
        state, obs, dreward = self.env.reset(self.trim)
        state_prev = None
        action_prev = None
        self.F = self.incremental_model.F
        self.G = self.incremental_model.G
        
        # Start (online) training loop, single episode
        for i in (bar := tqdm(range(self.num_timesteps))):
            bar.set_description("Training agent")
            
            # Action and state update
            action = self.call(np.array([obs]))
            
            if self.t == 1000 and self.cg == True:
                self.env.cg_shift()
                
            if self.t == 1000 and self.steering_fail == True:
                self.env.reduced_steering()

            state_next, obs_next, dreward = self.env.step(action)
            dreward = dreward * self.reward_scale
            self.t += 1


            # Update networks
            self.learn_step(np.array([obs]), np.array([obs_next]), state, dreward, self.F, self.G)

            # Update model
            if i > 1:
                self.F, self.G = self.incremental_model.update_model(state - state_prev, state_next - state, action - action_prev)

            # Update samples
            obs = obs_next
            action_prev = action
            state_prev = state
            state = state_next

            # Logging
            self.actor_weights_history.append(self.actor.get_weights())
            self.critic_weights_history.append(self.critic.get_weights())

    @tf.function(experimental_relax_shapes=True)
    def learn_step(self, obs, obs_next, state, dreward, F, G):

        with tf.GradientTape(persistent=True) as tape:

            # Actor call
            action = self.actor(obs)

            # Critic call
            lmbda = self.critic(obs)
         
        
        dreward = tf.cast(dreward, tf.float32)
        G = tf.cast(G, tf.float32)
        F = tf.cast(F, tf.float32)
        # Actor loss
        actor_loss_grad = -(dreward + self.gamma_discount * lmbda) @ G
        actor_loss_grad = tape.gradient(
            action, self.actor.trainable_weights, output_gradients=actor_loss_grad
        )

        # Actor update
        self.actor_optimizer.apply_gradients(zip(actor_loss_grad, self.actor.trainable_weights))

        # Critic loss
        dpidx = self.actor.get_policy_state_grad(state[:,0],obs)
        dpidx = tf.cast(dpidx, tf.float32)
        
        td_err_ds = (dreward + self.gamma_discount * lmbda) @ (F + G @ dpidx) - lmbda
        critic_loss_grad = -td_err_ds
        critic_loss_grad = tape.gradient(
            lmbda, self.critic.trainable_weights, output_gradients=critic_loss_grad
        )

        # Critic update
        self.critic_optimizer.apply_gradients(zip(critic_loss_grad, self.critic.trainable_weights))

        del tape

        
