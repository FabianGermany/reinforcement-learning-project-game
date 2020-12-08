#STEP(1): Import libraries
#-----------------------------------------------------------------------------------------------------------------------
import gym
from gym.envs.registration import registry, register, make, spec #for custom registration/game version
import numpy as np
import pandas as pd
#import tensorflow as tf
#import torch
import matplotlib.pyplot as plt
import random
import time
from IPython.display import clear_output

#STEP(2): Import libraries
#-----------------------------------------------------------------------------------------------------------------------
#register own version; see here: https://stackoverflow.com/questions/52727233/how-can-i-register-a-custom-environment-in-openais-gym
#the rest is done here: C:\Users\xxxxxxx\AppData\Local\Programs\Python\Python37\Lib\site-packages\gym\envs\box2d\_init_-py: from gym.envs.box2d.car_racing_custom import CarRacingCustom
register(
    id='CarRacingCustom-v1',
    entry_point='gym.envs.box2d:CarRacingCustom',
    max_episode_steps=1000,
    reward_threshold=900,
)

#STEP(3): Settings
#-----------------------------------------------------------------------------------------------------------------------
#modes
#mode = "no-learning"
mode = "reinforcement-learning-q-learning"
#mode = "reinforcement-learning-deep-q-learning"
#mode_no_learning_pattern = "specific_action"
mode_no_learning_pattern = "random"


#STEP(4): Load Environment
#-----------------------------------------------------------------------------------------------------------------------
#env = gym.make('CarRacing-v0') #global original file
env = gym.make('CarRacingCustom-v1') #local edited file car_racing_custom


#STEP(5): Plot features and parameters
#-----------------------------------------------------------------------------------------------------------------------
print('**********************************')
print('Configuration')
print('**********************************')

#Action Space (for discrete space)
#-------------------------
# action_space_size = env.action_space.shape[0] #this works for discrete space
# print('Action Space Data Types for Steer, Accelerate, Brake:')
# print(env.action_space.dtype)
# print('Action Space Min Value for Steer, Accelerate, Brake:')
# print(env.action_space.low)
# print('Action Space Max Value for Steer, Accelerate, Brake:')
# print(env.action_space.high)
# print('Action Space Size')
# print(action_space_size)

#Action Space (for continuous space)
#-------------------------
action_space_size = env.action_space.n  # this works for continuous space
print('Action Space Size:')
print(action_space_size)

print('Action Space Types:')
print(env.all_discrete_actions)

#Observation Space (RGB image data)
#-------------------------
state_space_size = env.observation_space.shape[0]*env.observation_space.shape[1]*env.observation_space.shape[2]*(env.observation_space.high[0,0,0]-env.observation_space.low[0,0,0]+1) #96*96*3*256 =7077888 (0...255)
#state_space_size2 = env.observation_space.n #todo was das
#state_space_size3 =

print('Oberservation Space Size:')
print(state_space_size)
print('Observation Space Data Types:')
print(env.observation_space.dtype)
print('Observation Space Min Value:')
print(env.observation_space.low[0,0,0])
print('Observation Space Max Value:')
print(env.observation_space.high[0,0,0])
print('Observation Space Format:')
print(env.observation_space.shape)




#STEP(6): Learn and test
#-----------------------------------------------------------------------------------------------------------------------
if (mode == "no-learning"):
    print('**********************************')
    print('Start Game')
    print('**********************************')

    # some standard values for experimental purposesm #values for steer, gas, brake
    ACTION_FULL_THROTTLE = [0, 1, 0]
    ACTION_SOFT_ACCELERATION = [0, 0.5, 0]
    ACTION_FULL_BRAKING = [0, 0, 1]
    ACTION_SOFT_BRAKING = [0, 0, 0.5]
    ACTION_TURN_LEFT_HARD = [-1, 0, 0]
    ACTION_TURN_RIGHT_HARD = [1, 0, 0]
    ACTION_TURN_LEFT_SOFT = [-0.5, 0, 0]
    ACTION_TURN_RIGHT_SOFT = [0.5, 0, 0]
    ACTION_ACCELERATE_AND_TURN_LEFT_SOFT = [-0.5, 0.5, 0]
    ACTION_ACCELERATE_AND_TURN_RIGHT_SOFT = [0.5, 0.5, 0]
    ACTION_DECELERATE_AND_TURN_LEFT_SOFT = [-0.5, 0, 0.5]
    ACTION_DECELERATE_AND_TURN_RIGHT_SOFT = [0.5, 0, 0.5]
    ACTION_DO_NOTHING = [0, 0, 0]

    n_iterations_no_training = 6  # number of iterations for casual
    duration_of_sim = 400  # duration of simulation in ms

    for i_episode in range(n_iterations_no_training):
        observation = env.reset()
        for t in range(duration_of_sim):
            env.render() #render one frame
            if(mode_no_learning_pattern == "specific_action"):
                action = ACTION_ACCELERATE_AND_TURN_RIGHT_SOFT  # specific action
            elif (mode_no_learning_pattern == "random"):
                action = env.action_space.sample()  # take a random action;
            observation, reward, done, info = env.step(action) # step delivers: observation (=state), reward, done and info
            print("Step {}:".format(t))
            print("Action: {}".format(action))
            print("Observation/State: \n{}".format(observation))
            print("Reward: {}".format(reward))
            print("Done: {}".format(done))
            print("Info: {}".format(info))
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()

    print('**********************************')
    print("Done.\n")
    print('**********************************')





elif (mode == "reinforcement-learning-q-learning"):
    print('**********************************')
    print('Training Agent')
    print('**********************************')

    n_iterations_training = 50  # number of iterations for reinforcement testing
    duration_of_sim = 400  # duration of simulation in ms

    gamma = 0.8  # future reward
    alpha = 0.3  # learning rate
    epsilon = 0.1 #todo

    q_table = np.zeros((state_space_size, action_space_size)) #initialize q-table with zeros
    print('Q table format:')
    print(q_table.shape)
    print('Q table size:')
    print(q_table.shape[0]*q_table.shape[1]) # 96*96*3*256*13 = 92012544 states
    #observation/state space is already discrete (RGB values) ? todo

    for i_episode in range(n_iterations_training):
        print("Starting Iteration {}".format(i_episode))
        observation = env.reset()
        reward = 0
        done = False

        for t in range(duration_of_sim):
            print("Starting Step {0} in Iteration {1}".format(t,i_episode))
            env.render()
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                #action = np.argmax(q_table[observation,:])  # Exploit learned values
                action_flattened_data = np.argmax(a=q_table[observation,:])  # Exploit learned values; check for current observation the action with maximum value and deliver the col (=action)
                action = action_flattened_data%q_table.shape[1] #first was flattened data, this gives really the col number form 0-12

            print("Action: {}".format(action))
            # Take action
            next_observation, reward, done, info = env.step(action) # step delivers: observation (=state), reward, done and info

            # Recalculate
            q_value = q_table[observation, action]
            max_value = np.max(q_table[next_observation])
            new_q_value = (1 - alpha) * q_value + alpha * (reward + gamma * max_value)

            # Update Q-table
            q_table[observation, action] = new_q_value
            observation = next_observation

            # if (duration_of_sim + 1) % 100 == 0:
            #     clear_output(wait=True)
            #     print("Episode: {}".format(duration_of_sim + 1))
            #     env.render()

            print("Observation/State: \n{}".format(observation))
            print("Reward: {}".format(reward))
            print("Done: {}".format(done))
            print("Info: {}".format(info))
            print("Q table: {}" .format(q_table))

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()

    print('**********************************')
    print("Training is done!\n")
    #todo print q-table here to see result...

    print('**********************************')





    print('**********************************')
    print('Testing/Evaluating performance of agent after Q-learning')
    print('**********************************')

    n_iterations_testing = 25  # number of iterations for reinforcement training
    penalties_total = 0
    timesteps_total = 0

    for i_episode in range(n_iterations_testing):
        observation = env.reset()
        timesteps = 0
        penalties = 0
        reward = 0

        terminated = False
        while not terminated:
            #action = np.argmax(q_table[observation,:])
            action_flattened_data = np.argmax(a=q_table[observation,
                                                :])  # Exploit learned values; check for current observation the action with maximum value and deliver the col (=action)
            action = action_flattened_data % q_table.shape[
                1]  # first was flattened data, this gives really the col number form 0-12
            observation, reward, terminated, info = env.step(action)

            if reward <= 0:
                penalties += 1
            timesteps += 1

        penalties_total += penalties
        timesteps_total += timesteps

    print('**********************************')
    print("Testing is done.\n")
    print('**********************************')


    print("**********************************")
    print("Results after {} iterations:".format(n_iterations_testing))
    print("**********************************")
    print("Timesteps per iteration: {}".format(timesteps_total / n_iterations_testing))
    print("Penalties per iteration: {}".format(penalties_total / n_iterations_testing))


    #inspired by https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/


elif (mode == "reinforcement-learning-deep-q-learning"):
        print("not yet")


#For discrete solution there are e.g. 5 potential actions todo:
#1: Forward/Accelerate
#2: Backward/Decelerate/Brake
#3: Steer to left
#4: Steer to right
#5: do nothing



#input("Press enter to exit this program.")
