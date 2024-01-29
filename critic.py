import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Flatten
import numpy as np

class Critic(keras.Model):


    def __init__(self, env, parameters):
        super().__init__()

        self.input_dimension = 1
        self.output_dimension = np.size(env.state,0)

        self.critic = None
        self.neurons = parameters["neurons"]

        self.kernel_initializer = keras.initializers.truncated_normal(mean = 0.0, stddev = parameters["std_init"], seed=parameters["seed"])
  
        self.setup()
        self.build(input_shape=(None, self.input_dimension))

    def setup(self):
        
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
            activation=None,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
        )(output)

        self.critic = keras.Model(inputs=input, outputs=output)

    def call(self, obs):

        return self.critic(obs)


