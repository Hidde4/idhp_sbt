import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Flatten
import numpy as np 

class Actor(keras.Model):

    def __init__(self, env, parameters):
        super().__init__()
        self.env = env

        self.input_dimension = 1
        self.output_dimension = 1

        self.actor = None
        self.neurons = parameters["neurons"]

        self.kernel_initializer = keras.initializers.truncated_normal(mean = 0.0, stddev = parameters["std_init"], seed=parameters["seed"])
  
        self.setup_actor()
        self.build(input_shape=(None, self.input_dimension))

    def setup_actor(self):
    

        input = keras.Input(shape=(self.input_dimension,))
        output = Flatten()(input)

        output = Dense(
            self.neurons,
            activation=tf.nn.tanh,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
        )(output)

 
        output = Dense(
            self.output_dimension,
            activation=tf.nn.tanh,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
        )(output)

        self.actor = keras.Model(inputs=input, outputs=output)

    def call(self, obs):

        action = self.actor(obs)

        return action

    @tf.function(experimental_relax_shapes=True)
    def get_policy_state_grad(self, state, obs):

        with tf.GradientTape(persistent=True) as state_tape:
            state_tape.watch(state)
            
            obs = self.env.get_obs(state)[np.newaxis]
            
            action = self(obs)
            
            action_nodes = tf.split(action, self.output_dimension, axis=1) 
        
        grads = []
        for i in range(len(action_nodes)):
            grads.append(state_tape.gradient(action_nodes[i], state))
       
        del state_tape
        dpidx = tf.stack(grads, axis=0)
        
        return dpidx
