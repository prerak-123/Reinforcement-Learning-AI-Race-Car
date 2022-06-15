import numpy as np
import matplotlib.pyplot as plt

REPAIR_COST = -20
PROFIT = 100
MAINTENANCE = 70

#Small changes only
class PlantModel:
    def __init__(self):
        self.age = 0
        self.condition = 1
        self.fail_probability = max(0.02 + 0.05*self.age, 1 - self.condition)
        
    def reset(self):
        self.age = 0
        self.condition = 1
        self.fail_probability = max(0.02 + 0.05*self.age, 1 - self.condition)
    
    def update(self, action):
        if (action == 0):
            
            if np.random.random() < self.fail_probability:
                reward = REPAIR_COST - 5*self.age
                self.reset()
                return (reward, self.age, self.condition)
            
            else:
                reward = PROFIT - 5*self.age
                self.age += 1
                self.condition = self.condition - 0.1 * np.random.random()
                self.fail_probability = max(0.02 + 0.05*self.age, 1 - self.condition)
                return (reward, self.age, self.condition)
        
        if (action == 1):
            self.reset()
            
            return (MAINTENANCE, self.age, self.condition)

ITERATIONS = 1000
GAMMA = 0.1
LEARNING_RATE = 0.2
epsilon = 0.8
DELTA_EPSILON = epsilon / 750

q_table = np.zeros(shape=(20, 51, 2))
plant = PlantModel()
plant_age = 0
plant_condition = 50
reward_arr = []
avg_reward_arr = []

for iteration in range(ITERATIONS):
    
    if np.random.random() < epsilon:
        action = np.random.randint(0, 2)
    
    else:
        action = np.argmax(q_table[plant_age][plant_condition])
    
    reward, new_plant_age, new_plant_condition = plant.update(action)
    new_plant_condition = int(50*new_plant_condition)
    
    reward_arr.append(reward)
    avg_reward_arr.append(np.sum(reward_arr) / (iteration + 1))
    
    if(new_plant_age != 0):
        q_table[plant_age][plant_condition][action] = (1 - LEARNING_RATE)*(q_table[plant_age][plant_condition][action]) + LEARNING_RATE*(reward + GAMMA*np.max(q_table[new_plant_age][new_plant_condition]))
    
    else:
        q_table[plant_age][plant_condition][action] = (1 - LEARNING_RATE)*(q_table[plant_age][plant_condition][action]) + LEARNING_RATE*(reward)
    
    plant_age = new_plant_age
    plant_condition = new_plant_condition
    epsilon -= DELTA_EPSILON
    

plt.plot(avg_reward_arr, label = "Average")
plt.legend()
plt.show()

np.save("Level_4_Q_Table", q_table)