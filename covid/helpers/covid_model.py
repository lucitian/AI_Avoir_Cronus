import numpy as np 

class SIRModel:
    def __init__(
        self, 
        transmission_rate: float, 
        recovery_rate: float,
        population: int,
        infecteds: int,
        recovereds: int,
        time_step: float,
    ):
        self.alpha = recovery_rate
        self.beta = transmission_rate 
        self.susceptibles = population - infecteds
        self.infecteds = infecteds 
        self.recovereds = recovereds
        self.dt = time_step

        # History for plotting 

        self.S_history = []
        self.I_history = []
        self.R_history = []

        self.daily_susceptibility_rate = []
        self.daily_infection_rate = []
        self.daily_recovery_rate = []

    def dS_dt(self):
        _ = self.beta * self.susceptibles * self.infecteds

        return _
    
    def dI_dt(self):
        _ = (self.beta * self.susceptibles * self.infecteds) - (self.dR_dt())

        return _
    
    def dR_dt(self):
        _ = self.alpha * self.infecteds 

        return _
    
    def S(self):
        self.susceptibles = self.susceptibles - (self.dt * self.dS_dt())
        self.daily_susceptibility_rate.append(self.dS_dt())
        self.S_history.append(self.susceptibles)
    
    def I(self):
        self.infecteds = self.infecteds + (self.dt * (self.dS_dt() - self.dR_dt()))
        self.daily_infection_rate.append(self.dI_dt())
        self.I_history.append(self.infecteds) 
    
    def R(self):
        self.recovereds = self.recovereds + (self.dt * self.dR_dt())
        self.daily_recovery_rate.append(self.dR_dt())
        self.R_history.append(self.recovereds)
    
    def R0(self):
        return 1 + (self.beta / self.alpha)
    
    def run(self, duration: int):
        timeframe = np.arange(start = 0, stop = duration - 1, step = self.dt)

        for _ in range(len(timeframe)):
            self.S()
            self.I()
            self.R()
        
        return timeframe