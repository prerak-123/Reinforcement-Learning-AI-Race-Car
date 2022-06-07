import numpy as np
import matplotlib.pyplot as plt

REPAIR_COST = -20
PROFIT = 100
MAINTENANCE = 70

class PlantModel:
    def __init__(self):
        self.age = 0
        self.condition = 1
        self.fail_probability = max(0.2 + 0.05*self.age, 1 - self.condition)
        
    def reset(self):
        self.age = 0
        self.condition = 1
        self.fail_probability = max(0.2 + 0.05*self.age, 1 - self.condition)
    
    def update(self, action):
        if (action == 0):
            
            if np.random.random() < self.fail_probability:
                self.reset()
                return (REPAIR_COST, self.age, self.condition)
            
            else:
                self.age += 1
                self.condition = self.condition - 0.1 * np.random.random()
                self.fail_probability = max(0.2 + 0.05*self.age, 1 - self.condition)
                return (PROFIT, self.age, self.condition)
        
        if (action == 1):
            self.reset()
            
            return (MAINTENANCE, self.age, self.condition)



