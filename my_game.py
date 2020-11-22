#STEP(1): Import libraries
#-----------------------------------------------------------------------------------------------------------------------
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#STEP(2): Declare Parameters
#-----------------------------------------------------------------------------------------------------------------------
duration_of_sim = 1000; #duration of simulation in ms
n_iterations = 3; #number of iterations


#STEP(3): Learn and test
#-----------------------------------------------------------------------------------------------------------------------
env = gym.make('CarRacing-v0')
for i_episode in range(n_iterations):
    observation = env.reset()
    for t in range(duration_of_sim):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action) # take a random action; step delivers: observation, reward, done and info
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()


#There are 5 potential actions:
#1: Forward/Accelerate
#2: Backward/Decelerate/Brake
#3: Steer to left
#4: Steer to right
#5: do nothing



#input("Press enter to exit this program.")
