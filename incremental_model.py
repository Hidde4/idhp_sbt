import numpy as np

class IncrementalModel:

    def __init__(self, env, parameters):
        self.env = env
        self.gamma = parameters["gamma_model"]
        self.cov_init = parameters["cov_init"]
       
        self.X = np.zeros((np.size(self.env.state)+1,1))
        
        self.Theta = np.zeros((np.size(self.X,0), np.size(self.env.state,0)))
        self.F = self.Theta[:np.size(self.env.state,0),:].T
        self.G = self.Theta[np.size(self.env.state,0):,:].T

        self.Cov = self.cov_init*np.identity(np.size(self.X))
        
        self.epsilon = np.zeros((1,np.size(self.env.state,0)))
       

    def update_model(self, state, state_next, action):
    
        self.X[:np.size(state,0)] = state
        
        self.X[np.size(state,0):] = np.array([action])[0]

        state_predict = (self.X.T @ self.Theta)
        
        self.epsilon = state_next.T - state_predict
        
        self.Theta = self.Theta + ((self.Cov @ self.X) / (self.gamma + self.X.T @ self.Cov @ self.X)) @ self.epsilon

        self.Cov = 1/self.gamma*(self.Cov - (self.Cov @ self.X @ self.X.T @ self.Cov) / (self.gamma + self.X.T @ self.Cov @ self.X))

        self.F = self.Theta[:np.size(state,0),:].T
        self.G = self.Theta[np.size(state,0):,:].T

        return self.F, self.G
    
    