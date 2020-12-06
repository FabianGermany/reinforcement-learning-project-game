#STEP(1): Import libraries
#-----------------------------------------------------------------------------------------------------------------------
import gym
from gym.envs.registration import registry, register, make, spec #for custom registration/game version

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#register own version; see here: https://stackoverflow.com/questions/52727233/how-can-i-register-a-custom-environment-in-openais-gym
#the rest is done here: C:\Users\xxxxxxx\AppData\Local\Programs\Python\Python37\Lib\site-packages\gym\envs\box2d\_init_-py: from gym.envs.box2d.car_racing_custom import CarRacingCustom
register(
    id='CarRacingCustom-v1',
    entry_point='gym.envs.box2d:CarRacingCustom',
    max_episode_steps=1000,
    reward_threshold=900,
)


#STEP(2): Declare Parameters
#-----------------------------------------------------------------------------------------------------------------------
duration_of_sim = 1000; #duration of simulation in ms
n_iterations = 3; #number of iterations

#some standard values for experimental purposes
action_full_throttle = [0, 1, 0]
action_full_braking = [0, 0, 1]
action_turn_left_hard = [-1, 0, 0]
action_turn_right_hard = [1, 0, 0]
action_turn_left_soft = [-0.5, 0, 0]
action_turn_right_soft = [0.5, 0, 0]
action_accelerate_and_turn_left_soft = [-0.5, 0.5, 0]
action_accelerate_and_turn_right_soft = [0.5, 0.5, 0]

env = gym.make('CarRacing-v0') #global original file
#env = gym.make('CarRacingCustom-v1') #local edited file car_racing_custom

print('**********************************')
print('Configuration')
print('**********************************')

#Action Space
#-------------------------
print('Action Space Data Types for Steer, Accelerate, Brake:')
print(env.action_space.dtype)
print('Action Space Min Value for Steer, Accelerate, Brake:')
print(env.action_space.low)
print('Action Space Max Value for Steer, Accelerate, Brake:')
print(env.action_space.high)

#Observation Space
#-------------------------
print('Observation Space Data Types:')
print(env.observation_space.dtype)
print('Observation Space Min Value:')
print(env.observation_space.low[0,0,0])
print('Observation Space Max Value:')
print(env.observation_space.high[0,0,0])
print('Observation Space Format:')
print(env.observation_space.shape)

#STEP(3): Learn and test
#-----------------------------------------------------------------------------------------------------------------------

print('**********************************')
print('Start Game')
print('**********************************')
for i_episode in range(n_iterations):
    observation = env.reset()
    for t in range(duration_of_sim):
        env.render() #render one frame
        print('Current Obvervation:')
        print(observation)
        #action = env.action_space.sample()  #take a random action;
        # action = action_accelerate_and_turn_right_soft #specific action
        action = env.action_space.sample()
        print('Current Action:')
        print(action)
        observation, reward, done, info = env.step(action) # step delivers: observation, reward, done and info
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
