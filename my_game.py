#STEP (1): Import libraries
#-----------------------------------------------------------------------------------------------------------------------
import gym
from gym.envs.registration import registry, register, make, spec #for custom registration/game version
import numpy as np
import pandas as pd
#import tensorflow as tf
#import torch
#import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import matplotlib.pyplot as plt
import random
import time
from IPython.display import clear_output
import h5py #saving big data (e.g. q learning table)

#STEP (2): Register custom game
#-----------------------------------------------------------------------------------------------------------------------
#register own version; see here: https://stackoverflow.com/questions/52727233/how-can-i-register-a-custom-environment-in-openais-gym
#the rest is done here: C:\Users\xxxxxxx\AppData\Local\Programs\Python\Python37\Lib\site-packages\gym\envs\box2d\_init_-py: from gym.envs.box2d.car_racing_custom import CarRacingCustom
register(
    id='CarRacingCustom-v1',
    entry_point='gym.envs.box2d:CarRacingCustom',
    max_episode_steps=1000,
    reward_threshold=900,
)

#STEP (3): Choosing mode and general settings
#-----------------------------------------------------------------------------------------------------------------------
#modes
# mode = "no-learning"
mode = "reinforcement-learning-q-learning"
# mode = "reinforcement-learning-deep-q-learning"
#mode_no_learning_pattern = "specific_action"
mode_no_learning_pattern = "random"
render_active = True
#render_active = False
reuse_improved_q_table = False
reuse_improved_agent = False



#STEP (4): Define reusable functions
#-----------------------------------------------------------------------------------------------------------------------

#get single pixels from whole image (better here than changing settings in gym files)
def get_nine_pixels(image):
    q11 = image[16, 16, 1]  # only green channel: third array arg. is 1 [0:2]
    q12 = image[16, 48, 1]
    q13 = image[16, 80, 1]
    q21 = image[48, 16, 1]
    q22 = image[48, 48, 1]
    q23 = image[48, 80, 1]
    q31 = image[80, 16, 1]
    q32 = image[80, 48, 1]
    q33 = image[80, 80, 1]
    nine_pixels = [q11, q12, q13, q21, q22, q23, q31, q32, q33]
    return nine_pixels

def discretize_RGB_val(list_of_256_value): #from 0...255 RGB value to [0,1,2,3]
    list_of_discretized_value = [None] * len(list_of_256_value)  # initialize list
    for i in range(len(list_of_256_value)):
        if(list_of_256_value[i]<64):
            list_of_discretized_value[i] = 0
        elif(list_of_256_value[i]<128):
            list_of_discretized_value[i] = 1
        elif(list_of_256_value[i]<192):
            list_of_discretized_value[i] = 2
        else: #list_of_256_value[i]<256
            list_of_discretized_value[i] = 3
    return list_of_discretized_value


#STEP (5): Load Environment
#-----------------------------------------------------------------------------------------------------------------------
#env = gym.make('CarRacing-v0') #global original file
env = gym.make('CarRacingCustom-v1') #local edited file car_racing_custom


#STEP (6): Plot features and parameters
#-----------------------------------------------------------------------------------------------------------------------
print('**********************************')
print('Configuration')
print('**********************************')

#Action Space (for discrete  space)
#-------------------------
action_space_size = env.action_space.n
print('Action Space Size:')
print(action_space_size) #13
print('Action Space Types:')
print(env.all_discrete_actions)

#Observation Space (RGB image data); kind continuous because from 0...255 (but not float)
#-------------------------
#state_space_size = env.observation_space.shape[0]*env.observation_space.shape[1]*env.observation_space.shape[2]*(env.observation_space.high[0,0,0]-env.observation_space.low[0,0,0]+1) #96*96*3*256 =7077888 (0...255) #this is wrong
#state_space_size = env.observation_space.shape[0]*env.observation_space.shape[1]*env.observation_space.shape[2] #=27648 #env.reset() automatically gives 27648 back (96*96*3)
state_space_size = 9 #because of the function get_nine_pixels we only have 9 values to observe and not 96*96*3=27648

#Nnumber of potential combinations
state_space_size_combinations = 262144 # /4**9 / 4^9 =262.144 #9 pixel version with use of discretize_RGB_val
#state_space_size_combinations = 256^27648 #full image version


print('Oberservation Space Size:')
print(state_space_size)
#print('Observation Space Data Types:')
#print(env.observation_space.dtype)
#print('Observation Space Min Value:')
#print(env.observation_space.low[0,0,0])
#print('Observation Space Max Value:')
#print(env.observation_space.high[0,0,0])
#print('Observation Space Format:')
#print(env.observation_space.shape)
print('Number of Observation Combinations:')
print(state_space_size_combinations)
print('Number of Action/Observation Combinations:')
print(state_space_size_combinations*action_space_size)

#STEP (7): Perform the desired driving algorithm
#-----------------------------------------------------------------------------------------------------------------------

#######################################################################################################################
#No learning
#######################################################################################################################
if (mode == "no-learning"):
    print('**********************************')
    print('Start Game')
    print('**********************************')

    # some standard values for experimental purposes #values for steer, gas, brake
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
        observation_focus = discretize_RGB_val(get_nine_pixels(observation)) #only use 9 pixels; and discretize values
        for t in range(duration_of_sim):
            if(render_active): env.render() #render one frame
            if(mode_no_learning_pattern == "specific_action"):
                action = ACTION_ACCELERATE_AND_TURN_RIGHT_SOFT  # specific action
            elif (mode_no_learning_pattern == "random"):
                action = env.action_space.sample()  # take a random action;
            observation, reward, done, info = env.step(action) # step delivers: observation (=state), reward, done and info
            observation_focus = discretize_RGB_val(get_nine_pixels(observation)) #only use 9 pixels; and discretize values

            #limit observation to single pixels
            print("Step {}:".format(t))
            print("Action: {}".format(action))
            print("Observation/State: \n{}".format(observation_focus))
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


#######################################################################################################################
#Q-Learning
#######################################################################################################################
elif (mode == "reinforcement-learning-q-learning"):
    print('**********************************')
    print('Training Agent')
    print('**********************************')

    n_iterations_training = 50  # number of iterations for reinforcement testing
    duration_of_sim = 600  # duration of simulation in ms

    gamma = 0.8  # future reward
    alpha = 0.3  # learning rate
    epsilon = 0.1 #todo

    #maybe use former q_table and optimize it
    if(reuse_improved_q_table):
        file_q_table_input = h5py.File("stored_agent/q-learning-table.hdf5", "r")
        q_table = file_q_table_input.get('dataset_learnt')
        file_q_table_input.close()
    else:
        q_table = np.zeros((state_space_size_combinations, action_space_size)) #initialize q-table with zeros


    print('Q table format:')
    print(q_table.shape)
    print('Q table size / amount of combinations/cells:')
    print(q_table.shape[0]*q_table.shape[1])

    for i_episode in range(n_iterations_training):
        print("Starting Iteration {}".format(i_episode))
        observation = env.reset()
        observation_focus = discretize_RGB_val(get_nine_pixels(observation))  # only use 9 pixels; and discretize values
        reward = 0
        done = False

        for t in range(duration_of_sim):
            print("Starting Step {0} in Iteration {1}".format(t,i_episode))
            if(render_active): env.render()
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:

                action = np.argmax(q_table[observation_focus,:])  # Exploit learned values
                #action_flattened_data = np.argmax(a=q_table[observation_focus,:])  # Exploit learned values; check for current observation the action with maximum value and deliver the col (=action)
                #action = action_flattened_data%q_table.shape[1] #first was flattened data, this gives really the col number form 0-12; todo bin mir nicht sicher ob das so stimmt

            print("Action: {}".format(action))
            # Take action
            next_observation, reward, done, info = env.step(action) # step delivers: observation (=state), reward, done and info
            next_observation_focus = discretize_RGB_val(get_nine_pixels(next_observation)) #only use 9 pixels; and discretize values


            # Recalculate
            q_value = q_table[observation_focus, action]
            max_value = np.max(q_table[next_observation_focus])
            new_q_value = (1 - alpha) * q_value + alpha * (reward + gamma * max_value)

            # Update Q-table
            q_table[observation_focus, action] = new_q_value
            observation_focus = next_observation_focus

            # if (duration_of_sim + 1) % 100 == 0:
            #     clear_output(wait=True)
            #     print("Episode: {}".format(duration_of_sim + 1))
            #     env.render()

            print("Observation/State: \n{}".format(observation_focus))
            print("Reward: {}".format(reward))
            print("Done: {}".format(done))
            print("Info: {}".format(info))
            print("Q table: {}" .format(q_table))

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break



    env.close()

    # save q table as hdf5 file
    file_q_table_output = h5py.File("stored_agent/q-learning-table.hdf5", "w")
    file_q_table_output.create_dataset('dataset_learnt', data=q_table)
    file_q_table_output.close()


    print('**********************************')
    print("Training is done!\n")
    print('**********************************')





    print('**********************************')
    print('Testing/Evaluating performance of agent after Q-learning')
    print('**********************************')

    n_iterations_testing = 5  # number of iterations for reinforcement training
    penalties_total = 0
    timesteps_total = 0

    for i_episode in range(n_iterations_testing):
        observation = env.reset()
        observation_focus = discretize_RGB_val(get_nine_pixels(observation))  # only use 9 pixels; and discretize values
        timesteps = 0
        penalties = 0
        reward = 0

        terminated = False
        while not terminated:
            action = np.argmax(q_table[observation,:])
            #action_flattened_data = np.argmax(a=q_table[observation_focus,:])  # Exploit learned values; check for current observation the action with maximum value and deliver the col (=action)
            #action = action_flattened_data % q_table.shape[1]  # first was flattened data, this gives really the col number form 0-12; todo bin mir nicht sicher ob das so stimmt

            print("Action: {}".format(action))

            observation, reward, terminated, info = env.step(action)
            observation_focus = discretize_RGB_val(get_nine_pixels(observation)) #only use 9 pixels; and discretize values


            print("Observation/State: \n{}".format(observation_focus))
            print("Reward: {}".format(reward))
            print("Done: {}".format(done))
            print("Info: {}".format(info))

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


#######################################################################################################################
#DQN Learning
#######################################################################################################################
elif (mode == "reinforcement-learning-deep-q-learning"):
        n_iterations_deep_q = 20
        duration_of_sim = 600

        print('**********************************')
        print('Creating Deep Q CNN Agent')
        print('**********************************')

        # https: // keon.github.io / deep - q - learning /, https://github.com/keon/deep-q-learning/blob/master/dqn.py
        class DQNAgent:
            def __init__(self, state_size, action_size):
                self.state_size = state_size
                self.action_size = action_size
                self.memory = deque(maxlen=2000)
                self.gamma = 0.95  # discount rate
                self.epsilon = 1.0  # exploration rate
                self.epsilon_min = 0.01
                self.epsilon_decay = 0.995
                self.learning_rate = 0.001
                self.model = self._build_model()

            def _build_model(self):
                # Neural Net for Deep-Q learning Model
                model = Sequential()
                model.add(Dense(24, input_dim=self.state_size, activation='relu')) #Input Layer: Image with the pixels: todo 96x96x3
                model.add(Dense(24, activation='relu')) #hidden layer #todo 24 anpassen..., siehe andere paper
                model.add(Dense(self.action_size, activation='linear')) #output layer
                model.compile(loss='mse',
                              optimizer=Adam(lr=self.learning_rate))
                return model

            #store and load experience
            def memorize(self, state, action, reward, next_state, done):
                self.memory.append((state, action, reward, next_state, done))

            def act(self, state):
                if np.random.rand() <= self.epsilon:
                    return random.randrange(self.action_size)
                act_values = self.model.predict(state)
                return np.argmax(act_values[0])  # returns action

            #train net with experiences stared in memory
            def replay(self, batch_size):
                minibatch = random.sample(self.memory, batch_size)
                for state, action, reward, next_state, done in minibatch:
                    target = reward
                    if not done:
                        target = reward + self.gamma * \
                                 np.amax(self.model.predict(next_state)[0])
                    target_f = self.model.predict(state)
                    target_f[0][action] = target
                    self.model.fit(state, target_f, epochs=1, verbose=0) # model fit
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

            def load(self, name):
                self.model.load_weights(name)

            def save(self, name):
                self.model.save_weights(name)


        # initialize gym environment and the agent
        #agent = DQNAgent(env)
        agent = DQNAgent(state_space_size_combinations, action_space_size) #todo oder _combinations...

        if(reuse_improved_agent):
            agent.load("stored_agent/dqn.h5")

        print('Deep Q CNN Agent created.')



        print('**********************************')
        print('Training Deep Q CNN Agent')
        print('**********************************')



        # Iterate the game
        for i_episode in range(n_iterations_deep_q):
            print("Starting Iteration {}".format(i_episode))
            state = env.reset()
            state_focus = discretize_RGB_val(get_nine_pixels(state))  # only use 9 pixels; and discretize values
            state_focus = np.reshape(state_focus, [1, state_space_size_combinations]) #todo warum nicht von 0 bis n-1 (bei dem anderen auch)
            reward = 0
            done = False
            batch_size = 32


            # time_t represents each frame of the game
            # Our goal is to keep the pole upright as long as possible until score of 500
            # the more time_t the more score
            for t in range(duration_of_sim):
                if (render_active): env.render()  # render one frame

                # Decide action
                action = agent.act(state)

                print("Action: {}".format(action))

                next_state, reward, done, info = env.step(action)   # Advance the game to the next frame based on the action.
                next_state_focus = discretize_RGB_val(get_nine_pixels(next_state))  # only use 9 pixels; and discretize values
                next_state_focus = discretize_RGB_val(get_nine_pixels(next_state_focus))  # only use 9 pixels; and discretize values

                next_state_focus = np.reshape(next_state_focus, [1, state_space_size_combinations]) #todo warum nicht von 0 bis n-1 (bei dem anderen auch)

                # memorize the previous state, action, reward, and done
                agent.memorize(state, action, reward, next_state_focus, done)

                # make next_state the new current state for the next frame.
                state_focus = next_state_focus

                print("Observation/State: \n{}".format(state_focus))
                print("Reward: {}".format(reward))
                print("Done: {}".format(done))
                print("Info: {}".format(info))

                if done:
                    print("episode: {}/{}, score: {}"
                          .format(i_episode, n_iterations_deep_q, t))
                    break

            # train the agent with the experience of the episode
            agent.replay(batch_size)

            #store model
            if (reuse_improved_agent):
                if i_episode % 10 == 0:
                    agent.save("stored_agent/dqn.h5")

        print('Deep Q CNN Agent trained.')




        print('**********************************')
        print('Test Deep Q CNN Agent')
        print('**********************************')


        print('Deep Q CNN Agent tested.')



#input("Press enter to exit this program.")
