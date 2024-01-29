import numpy as np 
import gym 
from gym.spaces import Box

class SolarBoat(gym.Env):

    def __init__(self, task):
        
        # Time
        self.task = task
        self.dt = task.dt
        self.t  = 0

        # Constants
        self.rho = 1000 # Water density [kg/m^3]
        self.g   = 9.81 # Gravitational acceleration [m/s^2]
        self.V   = 10   # Velocity [m/s]
        self.m   = 167  # Boat mass [kg]

        # Geometry 
        self.c_sf   = 0.089  # Front strut chord [m]
        self.c_sr   = 0.177  # Rear strut chord [m]
        self.d_z_s  = 0.9    # Center of mass to effective end of strut [m]
        self.d_z_m  = 0.5    # Center of mass flying height above true water line [m]
        self.d_z_id = 0.2    # Offset betweeen idealized and true waterline [m]
        self.S_wf   = 0.0319 # Surface area of front wing [m^2]
        self.S_wr   = 0.0681 # Surface area of rear wing [m^2]
        self.d_x_sf = 2.53   # Front strut distance of center of pressure to center of mass [m] 
        self.d_x_sr = 1.38   # Rear strut distance of center of pressure to center of mass [m]
        self.b_wf   = 0.708  # Front wing span [m] 
        self.b_wr   = 0.997  # Rear wing span [m] 

        # Hydrodynamic properties 
        self.C_L_a_w = 5.7  # Lift curve slope of the wings [1/rad]
        self.C_L_a_s = 6.67 # Lift curve slope of the struts [1/rad]

        # Mass moment of inertia [kg m^2]
        self.I_xx =  18.3 
        self.I_zz = 219.1
        self.I_xz = -2.9

        # Inertia factors:
        self.K_xx = self.I_xx/(self.I_xx*self.I_zz-self.I_xz**2)
        self.K_zz = self.I_zz/(self.I_xx*self.I_zz-self.I_xz**2)
        self.K_xz = self.I_xz/(self.I_xx*self.I_zz-self.I_xz**2)

        # Idealized flying height 
        self.d_z_fly = self.d_z_m + self.d_z_id

        # Submerged strut surface area's
        self.S_sf = self.c_sf*(self.d_z_s-self.d_z_fly) # Front strut 
        self.S_sr = self.c_sr*(self.d_z_s-self.d_z_fly) # Rear strut

        # Nominal lift of the wings 
        self.L_nom_wf = self.m*self.g*self.S_wf/(self.S_wf+self.S_wr)
        self.L_nom_wr = self.m*self.g*self.S_wr/(self.S_wf+self.S_wr)

        # Derivative coefficients 
        self.Y_v     = -0.5*self.rho*self.V*self.C_L_a_s*(self.S_sf+self.S_sr) 
        self.L_v     =  0.5*self.rho*self.V*self.C_L_a_s*(self.d_z_s+self.d_z_fly)/2*(self.S_sf+self.S_sr)
        self.N_v     =  0.5*self.rho*self.V*self.C_L_a_s*(-self.S_sf*self.d_x_sf+self.S_sr*self.d_x_sr)
        self.Y_phi   =  self.m*self.g
        self.Y_p     =  0.5*self.rho*self.V*self.C_L_a_s*(self.c_sf+self.c_sr)*0.5*(self.d_z_s**2-self.d_z_fly**2)
        self.L_p     = -0.5*self.rho*self.V*(self.C_L_a_w*1/16*(self.S_wf*self.b_wf**2+self.S_wr*self.b_wr**2)+self.C_L_a_s*(self.c_sf+self.c_sr)*1/3*(self.d_z_s**3-self.d_z_fly**3))
        self.N_p     =  0.5*self.rho*self.V*self.C_L_a_s*(self.c_sf*self.d_x_sf-self.c_sr*self.d_x_sr)*0.5*(self.d_z_s**2-self.d_z_fly**2)-1/16*1/self.V*(self.L_nom_wf*self.b_wf**2+self.L_nom_wr*self.b_wr**2)
        self.Y_r     =  0.5*self.rho*self.V*self.C_L_a_s*(-self.S_sf*self.d_x_sf+self.S_sr*self.d_x_sr)
        self.L_r     =  0.5*self.rho*self.V*self.C_L_a_s*(self.d_z_s+self.d_z_fly)/2*(self.S_sf*self.d_x_sf-self.S_sr*self.d_x_sr)+1/8*1/self.V*(self.L_nom_wf*self.b_wf**2+self.L_nom_wr*self.b_wr**2)
        self.N_r     = -0.5*self.rho*self.V*self.C_L_a_s*(self.S_sf*self.d_x_sf**2+self.S_sr*self.d_x_sr**2)
        self.Y_gamma =  0.5*self.rho*self.V**2*self.S_sf*self.C_L_a_s
        self.L_gamma = -0.5*self.rho*self.V**2*self.S_sf*self.C_L_a_s*(self.d_z_s+self.d_z_fly)/2
        self.N_gamma =  0.5*self.rho*self.V**2*self.S_sf*self.d_x_sf*self.C_L_a_s


        # State space system A matrix
        self.A = np.zeros((4,4))
        self.A[0,0] = self.Y_v/self.m
        self.A[2,0] = self.L_v*self.K_zz+self.N_v*self.K_xz
        self.A[3,0] = self.N_v*self.K_xx+self.L_v*self.K_xz
        self.A[0,1] = self.Y_phi/self.m
        self.A[0,2] = self.Y_p/self.m
        self.A[1,2] = 1
        self.A[2,2] = self.L_p*self.K_zz+self.N_p*self.K_xz
        self.A[3,2] = self.N_p*self.K_xx+self.L_p*self.K_xz
        self.A[0,3] = self.Y_r/self.m-self.V
        self.A[2,3] = self.L_r*self.K_zz+self.N_r*self.K_xz
        self.A[3,3] = self.N_r*self.K_xx+self.L_r*self.K_xz

        # State space system B matrix
        self.B = np.zeros((4,1))
        self.B[0] = self.Y_gamma/self.m
        self.B[2] = self.L_gamma*self.K_zz+self.N_gamma*self.K_xz
        self.B[3] = self.N_gamma*self.K_xx+self.L_gamma*self.K_xz

        # Action space
        self.action_space = Box(
            low=np.array([-np.deg2rad(15.0)]),
            high=np.array([np.deg2rad(15.0)]),
            dtype=np.float64,
        )

        # Initialization 
        self.state = np.zeros((4,1))
        self.state_history = np.zeros((np.size(self.state,0), len(task.t_range)))
        self.action_history = np.zeros((len(task.t_range)))
        self.obs = 0.0
        self.dreward = 0.0

    def reset(self,trim):
        # Reset to initial conditions
        if trim:
            self.state = np.zeros((4,1))
        else:
            self.state = np.random.uniform(-5/180*np.pi,5/180*np.pi,(4,1))    
        self.state_history = np.zeros((np.size(self.state,0), len(self.task.t_range)))
        self.action_history = np.zeros((len(self.task.t_range)))
        self.obs = 0.0
        self.dreward = 0.0

        return self.state, self.obs, self.dreward
        
    
    def step(self, action):
        # State update
        self.t = self.t+1
        low, high = self.action_space.low, self.action_space.high
        action = low + 0.5 * (action + 1.0) * (high - low)
        self.state = self.state + (self.A.dot(self.state)+self.B.dot(np.array(action)[0]))*self.dt
        self.state_history[:, self.t -1] = self.state[:,0]
        self.action_history[self.t-1] = np.array(action)[0]

        ref_error = (self.state[1]-self.task.bank_reference[self.t-1])[0]*180/np.pi
        
        self.obs = ref_error
        self.dreward = np.array([0.,-2.0*ref_error,0.,0.])
        
        return self.state, self.obs, self.dreward 

    def get_obs(self, state):
        ref_error = (state[1]-self.task.bank_reference[self.t-1])*180/np.pi
        
        return ref_error

    def cg_shift(self):
        # Geometry 
        self.d_x_sf = 2.03   # Front strut distance of center of pressure to center of mass [m] 
        self.d_x_sr = 1.88   # Rear strut distance of center of pressure to center of mass [m] 

        # Idealized flying height 
        self.d_z_fly = self.d_z_m + self.d_z_id

        # Submerged strut surface area's
        self.S_sf = self.c_sf*(self.d_z_s-self.d_z_fly) # Front strut 
        self.S_sr = self.c_sr*(self.d_z_s-self.d_z_fly) # Rear strut

        # Nominal lift of the wings 
        self.L_nom_wf = self.m*self.g*self.S_wf/(self.S_wf+self.S_wr)
        self.L_nom_wr = self.m*self.g*self.S_wr/(self.S_wf+self.S_wr)

        # Derivative coefficients 
        self.Y_v     = -0.5*self.rho*self.V*self.C_L_a_s*(self.S_sf+self.S_sr) 
        self.L_v     =  0.5*self.rho*self.V*self.C_L_a_s*(self.d_z_s+self.d_z_fly)/2*(self.S_sf+self.S_sr)
        self.N_v     =  0.5*self.rho*self.V*self.C_L_a_s*(-self.S_sf*self.d_x_sf+self.S_sr*self.d_x_sr)
        self.Y_phi   =  self.m*self.g
        self.Y_p     =  0.5*self.rho*self.V*self.C_L_a_s*(self.c_sf+self.c_sr)*0.5*(self.d_z_s**2-self.d_z_fly**2)
        self.L_p     = -0.5*self.rho*self.V*(self.C_L_a_w*1/16*(self.S_wf*self.b_wf**2+self.S_wr*self.b_wr**2)+self.C_L_a_s*(self.c_sf+self.c_sr)*1/3*(self.d_z_s**3-self.d_z_fly**3))
        self.N_p     =  0.5*self.rho*self.V*self.C_L_a_s*(self.c_sf*self.d_x_sf-self.c_sr*self.d_x_sr)*0.5*(self.d_z_s**2-self.d_z_fly**2)-1/16*1/self.V*(self.L_nom_wf*self.b_wf**2+self.L_nom_wr*self.b_wr**2)
        self.Y_r     =  0.5*self.rho*self.V*self.C_L_a_s*(-self.S_sf*self.d_x_sf+self.S_sr*self.d_x_sr)
        self.L_r     =  0.5*self.rho*self.V*self.C_L_a_s*(self.d_z_s+self.d_z_fly)/2*(self.S_sf*self.d_x_sf-self.S_sr*self.d_x_sr)+1/8*1/self.V*(self.L_nom_wf*self.b_wf**2+self.L_nom_wr*self.b_wr**2)
        self.N_r     = -0.5*self.rho*self.V*self.C_L_a_s*(self.S_sf*self.d_x_sf**2+self.S_sr*self.d_x_sr**2)
        self.Y_gamma =  0.5*self.rho*self.V**2*self.S_sf*self.C_L_a_s
        self.L_gamma = -0.5*self.rho*self.V**2*self.S_sf*self.C_L_a_s*(self.d_z_s+self.d_z_fly)/2
        self.N_gamma =  0.5*self.rho*self.V**2*self.S_sf*self.d_x_sf*self.C_L_a_s


        # State space system A matrix
        self.A = np.zeros((4,4))
        self.A[0,0] = self.Y_v/self.m
        self.A[2,0] = self.L_v*self.K_zz+self.N_v*self.K_xz
        self.A[3,0] = self.N_v*self.K_xx+self.L_v*self.K_xz
        self.A[0,1] = self.Y_phi/self.m
        self.A[0,2] = self.Y_p/self.m
        self.A[1,2] = 1
        self.A[2,2] = self.L_p*self.K_zz+self.N_p*self.K_xz
        self.A[3,2] = self.N_p*self.K_xx+self.L_p*self.K_xz
        self.A[0,3] = self.Y_r/self.m-self.V
        self.A[2,3] = self.L_r*self.K_zz+self.N_r*self.K_xz
        self.A[3,3] = self.N_r*self.K_xx+self.L_r*self.K_xz

        # State space system B matrix
        self.B = np.zeros((4,1))
        self.B[0] = self.Y_gamma/self.m
        self.B[2] = self.L_gamma*self.K_zz+self.N_gamma*self.K_xz
        self.B[3] = self.N_gamma*self.K_xx+self.L_gamma*self.K_xz
        return
    
    def reduced_steering(self):

        # State space system B matrix
        self.B = self.B*0.25
     
        return