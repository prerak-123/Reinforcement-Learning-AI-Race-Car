import numpy as np
import matplotlib.pyplot as plt

REPAIR_COST = -20
PROFIT = 100
MAINTENANCE = 70

class PlantModel:
    def __init__(self):
        self.fail_probability = 0
        self.age = 0
        
    def reset(self):
        self.fail_probability = 0
        self.age = 0
    
    def update(self, action):
        if (action == 0):
            
            if np.random.random() < self.fail_probability:
                reward = REPAIR_COST - 5*self.age
                self.age = 0
                self.fail_probability = 0
                return (reward, self.age)
            
            else:
                reward = PROFIT - 5*self.age
                self.age += 1
                self.fail_probability = self.age * 0.05
                return (reward, self.age)
        
        if (action == 1):
            self.age = 0
            self.fail_probability = 0
            
            return (MAINTENANCE, self.age)

ITERATIONS = 1000
GAMMA = 0.1
LEARNING_RATE = 0.2
epsilon = 0.8
DELTA_EPSILON = epsilon / 500

q_table = np.zeros(shape = (20, 2))   
plant_age = 0
reward_arr = []
avg_reward_arr = []

plant = PlantModel()

for i in range(ITERATIONS):
    if (np.random.random() < epsilon):
        action = np.random.randint(0, 2)
    else:
        action = np.argmax(q_table[plant_age])
    
    reward, new_plant_age = plant.update(action)
    
    reward_arr.append(reward)
    avg_reward_arr.append(np.sum(reward_arr) / (i + 1))
        
    q_table[plant_age][action] = (1 - LEARNING_RATE)*q_table[plant_age][action] + LEARNING_RATE * (reward + GAMMA * np.max(q_table[new_plant_age]))    
    
    plant_age = new_plant_age
    epsilon -= DELTA_EPSILON
   
plt.plot(avg_reward_arr, label = "Average")
plt.legend()
plt.show()

np.save("Level_3_Q_Table", q_table)