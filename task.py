import numpy as np 

class BankAngleTracking():
    """
    Bank angle tracking task 
    """

    def __init__(self, T, dt):
        self.T = T
        self.dt = dt
        self.t_range = np.arange(0,self.T,self.dt)
        self.bank_reference = self.set_bank_reference()
    
    def set_bank_reference(self):
        """
        Bank angle reference signal construction
        """

        amplitude = 2.5*np.pi/180 # Amplitude [rad]
        T1 = 5 # Period of first sine [s]
        T2 = 10  # Period of second sine [s]
        T3 = 2.5 # Period of third Sine [s]

        bank_reference = amplitude*(np.sin(2*np.pi*self.t_range/T1) + np.sin(2*np.pi*(self.t_range)/T2) + np.sin(2*np.pi*(self.t_range)/T3))

        return bank_reference