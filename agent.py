#STEP (1): Import libraries
#-----------------------------------------------------------------------------------------------------------------------
import gym
from gym.envs.registration import register#, registry, make, spec #for custom registration/game version
import numpy as np
import pandas as pd
#import tensorflow as tf
#import torch
#import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
#import matplotlib.pyplot as plt
import random
#import time
#from IPython.display import clear_output
#import h5py #saving and reloading big data (e.g. q learning table)
#import csv #saving and reloading big data (e.g. q learning table)

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
render_check_sometimes = False #only render sometimes
reuse_improved_q_table = True
reuse_improved_agent = False

duration_of_sim = 750 #duration of simulation in ms

#STEP (4): Define reusable functions
#-----------------------------------------------------------------------------------------------------------------------

#get single pixels from whole image (better here than changing settings in gym files); count from top left to buttom right
# only green channel: third array arg. is 1 [0:2]
# the three centers of 0...95 are about 16,30 and 80
def get_nine_pixels(image):
    q11 = image[80, 16, 1]
    q12 = image[72, 48, 1] #this is an exception cause image[80, 48, 1] would point to the car itself
    q13 = image[80, 80, 1]
    q21 = image[48, 16, 1]
    q22 = image[48, 48, 1]
    q23 = image[48, 80, 1]
    q31 = image[16, 16, 1]
    q32 = image[16, 48, 1]
    q33 = image[16, 80, 1]
    nine_pixels = [q11, q12, q13, q21, q22, q23, q31, q32, q33]
    return nine_pixels

#simplify 0...255 area to 0...3 area
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

#convert ndarray [0,0...,0] to decimal number
    # [0,0,0,0,0,0,0,0,0] --> Row 1
    # [0,0,0,0,0,0,0,0,1] --> Row 2
    # [0,0,0,0,0,0,0,0,2] --> Row 3
    # [0,0,0,0,0,0,0,0,3] --> Row 4
    # [0,0,0,0,0,0,0,1,0] --> Row 5
    # ...
    # [3,3,3,3,3,3,3,3,3] --> Row 5
    # is correct: we have state_space_size_combinations = 262144 combinations:
    # 262144[Base10] = 1000000000[Base4]
    # 262143[Base10] = 333333333[Base4]
def conv_base_4_ndarray_to_decimal_index (array):
    # to get the right row in the q learning table you need to convert the array [0,0,....,1] etc. to an index:
    # 1 merge the the 9 numbers to one number
    observation_as_one_int_base_4_as_string = str(array[0]) \
                                            + str(array[1])\
                                            + str(array[2])\
                                            + str(array[3])\
                                            + str(array[4])\
                                            + str(array[5])\
                                            + str(array[6])\
                                            + str(array[7])\
                                            + str(array[8])

    # 2 convert this number from base4 to base10 (decimal) = index
    observation_as_one_int_base_10 = int(observation_as_one_int_base_4_as_string,4) #int converts back to decimal
    index_row_q_table = observation_as_one_int_base_10 #this is the index
    return index_row_q_table

#combination of the three functions defined above: nine pixels, discretize values and convert ndarray to decimal index
def custom_observation_conversion(data):
    converted_data = conv_base_4_ndarray_to_decimal_index(discretize_RGB_val(get_nine_pixels(data)))
    return converted_data


#STEP (5): Load Environment
#-----------------------------------------------------------------------------------------------------------------------
#env = gym.make('CarRacing-v0') #global original file
env = gym.make('CarRacingCustom-v1') #local edited file car_racing_custom


#STEP (6): Print features and parameters
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
#STEP (7A): #No learning
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

    n_iterations_no_training = 35  # number of iterations for performing the actions
    cumulated_reward_added_up = 0
    average_reward_added_up = 0
    all_iterations_added_up = 0
    list_of_average_rewards = []

    for i_episode in range(n_iterations_no_training):
        print("Starting Iteration {}".format(i_episode))
        observation = env.reset()
        observation_focus = custom_observation_conversion(observation) #only use 9 pixels; and discretize values; and convert to decimal index
        reward = 0
        cumulated_reward = 0
        for t in range(duration_of_sim):
            print("Starting Step {0} in Iteration {1}".format(t,i_episode))
            if(render_active): env.render() #render one frame
            if(mode_no_learning_pattern == "specific_action"):
                action = ACTION_ACCELERATE_AND_TURN_RIGHT_SOFT  # specific action
            elif (mode_no_learning_pattern == "random"):
                action = env.action_space.sample()  # take a random action;
            observation, reward, done, info = env.step(action) # step delivers: observation (=state), reward, done and info
            observation_focus = custom_observation_conversion(observation) #only use 9 pixels; and discretize values; and convert to decimal index
            cumulated_reward += reward  # add current reward to cumulated reward

            #limit observation to single pixels
            print("Step {}:".format(t))
            #print("Action: {}".format(action))
            #print("Observation/State: \n{}".format(observation_focus))
            #print("Reward: {}".format(reward))
            print("Cumulated Reward: {:+0.2f}".format(cumulated_reward))
            #print("Done: {}".format(done))
            print("Info: {}".format(info))
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

        print("Cumulative reward: {:+0.2f}".format(cumulated_reward))
        print("Average reward per iteration: {:+0.3f}".format(cumulated_reward / duration_of_sim))
        cumulated_reward_added_up += cumulated_reward
        average_reward_added_up += (cumulated_reward / duration_of_sim)
        list_of_average_rewards = np.append(list_of_average_rewards, (cumulated_reward / duration_of_sim))


    env.close()

    print('**********************************')
    print("Done.\n")
    print('**********************************')

    print("Results after {} iterations:".format(n_iterations_no_training))
    print("**********************************")
    #print("Average cumulative reward {:+0.2f}".format(cumulated_reward_added_up / n_iterations_no_training))
    #print("Average reward per iteration for all iterations (average) {:+0.3f}".format(average_reward_added_up / n_iterations_no_training))

    list_of_average_rewards_rounded = ['%.3f' % elem for elem in list_of_average_rewards] #round list entries
    print("This list of all average rewards is:")
    print(', '.join(list_of_average_rewards_rounded)) #makes the print without quotes

    print("Average reward per iteration for all iterations (average) {:+0.3f}".format(np.mean(list_of_average_rewards)))
    print("Standard deviation: {:+0.3f}".format(np.std(list_of_average_rewards)))



#######################################################################################################################
#STEP (7B): #Q-Learning
#######################################################################################################################
elif (mode == "reinforcement-learning-q-learning"):
    print('**********************************')
    print('Training Agent')
    print('**********************************')

    # STEP (7B.1): #Parameters
    n_iterations_training = 0  # number of iterations for reinforcement training

    gamma = 0.5  # discount factor = future reward (0: short-term/greedy; 1: long-term)
    alpha = 0.4  # learning rate (0: learn nothing/just exploit prior knowledge; 1: ignore prior knowledge/focus on most recent information
    epsilon = 0.1 #used for epsilon-greedy algorithm (balance between explore and exploit; high value: explore; low value: exploit)
    # (epsilon might also be non-constant and change from high value to zero in the course of time...)

    # STEP (7B.2): #Load q-table
    #maybe use former q_table and optimize it: using hdf5 or csv
    if(reuse_improved_q_table):
        #file_q_table_input = h5py.File("stored_agent/q-learning-table.hdf5", "r")
        #q_table = file_q_table_input.get('dataset_learnt')
        #file_q_table_input.close()
        q_table = pd.read_csv('stored_agent/q-learning-table.csv',  header=None, sep=';')
        #q_table from csv is in pd-df format: reconvert to ndarray format
        q_table = q_table.to_numpy()
        print("Existing Q table has been loaded.\n")

    else:
        q_table = np.zeros((state_space_size_combinations, action_space_size)) #initialize q-table with zeros


    print('Q table format:')
    print(q_table.shape)
    print('Q table size / amount of combinations/cells:')
    print(q_table.shape[0]*q_table.shape[1])

    # STEP (7B.3): #Start outer loop
    for i_episode in range(n_iterations_training):
        print("Starting Iteration {}".format(i_episode))
        observation = env.reset()
        observation_focus = custom_observation_conversion(observation) #only use 9 pixels; and discretize values; and convert to decimal index
        reward = 0
        cumulated_reward = 0
        done = False

        # STEP (7B.4): #Start inner loop
        for t in range(duration_of_sim):
            print("Starting Step {0} in Iteration {1}".format(t,i_episode))
            if(render_check_sometimes):
                if(render_active):
                    if(t%(duration_of_sim/3)==0):#render only a couple of times per episode to see whether everything is fine or there is a bug like blackscreen
                        env.render()
            else: #NOT render_check_sometimes
                if(render_active):
                    env.render()

            # STEP (7B.5): #epsilon-greedy-algorithm
            if random.uniform(0, 1) < epsilon: #with probability epsilon:
                action = env.action_space.sample() # Explore action space (random action)
            else: #with probability 1-epsilon:
                action = np.argmax(q_table[observation_focus,:]) # Exploit learned values / take the best action from the Q table refering to the corresponding observation (greedy-action)

            # STEP (7B.6): #Take action
            #print("Action: {}".format(action))
            next_observation, reward, done, info = env.step(action) # step delivers: observation (=state), reward, done and info
            next_observation_focus = custom_observation_conversion(next_observation) #only use 9 pixels; and discretize values; and convert to decimal index

            cumulated_reward += reward #add current reward to cumulated reward

            # STEP (7B.7): #Recalculate and update Q-table
            q_value = q_table[observation_focus, action] #get current/old q-value from current/old table
            max_value = np.max(q_table[next_observation_focus]) #get the value of the action of current observation for best action
            new_q_value = (1 - alpha) * q_value + alpha * (reward + gamma * max_value) #update qvalue using bellman equation
            q_table[observation_focus, action] = new_q_value #update Q-table; use old state and action
            observation_focus = next_observation_focus #the new state will become the old one

            #print("Observation/State: \n{}".format(observation_focus))
            #print("Reward: {:+0.2f}".format(reward))
            print("Cumulated Reward: {:+0.2f}".format(cumulated_reward))
            #print("Done: {}".format(done))
            print("Info: {}".format(info))
            #print("Q table: {}" .format(q_table))

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

        # STEP (7B.8): #regularly save q table as hdf5 file or csv file
        if((i_episode%30==0) or (i_episode==n_iterations_training)):
            #file_q_table_output = h5py.File("stored_agent/q-learning-table.hdf5", "w")
            #file_q_table_output.create_dataset('dataset_learnt', data=q_table)
            #file_q_table_output.close()
            q_table_as_df = pd.DataFrame(q_table)
            q_table_as_df.to_csv('stored_agent/q-learning-table.csv',index=False, header=False, sep=';')
            print("Q table has been stored.\n")

    env.close()


    print('**********************************')
    print("Training is done!\n")
    print('**********************************')


    print('Testing/Evaluating performance of agent after Q-learning')
    print('**********************************')

    # STEP (7B.9): #Parameters for testing
    n_iterations_testing = 35  # number of iterations for reinforcement training
    cumulated_reward_added_up = 0
    average_reward_added_up = 0
    all_iterations_added_up = 0
    list_of_average_rewards = []

    # STEP (7B.10): #Start outer loop for testing
    for i_episode in range(n_iterations_testing):
        print("Iteration No. {}".format(i_episode))
        observation = env.reset()
        observation_focus = custom_observation_conversion(observation) #only use 9 pixels; and discretize values; and convert to decimal index
        reward = 0
        cumulated_reward = 0
        terminated = False

        # STEP (7B.11): #Start inner loop for testing
        for t in range(duration_of_sim):
            if (render_check_sometimes):
                if (render_active):
                    if (t % (duration_of_sim / 3) == 0):  # render only a couple of times per episode to see whether everything is fine or there is a bug like blackscreen
                        env.render()
            else:  # NOT render_check_sometimes
                if (render_active):
                        env.render()

            # STEP (7B.12): #Take action
            action = np.argmax(q_table[observation_focus, :]) #use the available qtable (dont learn anything new)
            #print("Action: {}".format(action))

            observation, reward, terminated, info = env.step(action)
            observation_focus = custom_observation_conversion(observation) #only use 9 pixels; and discretize values; and convert to decimal index

            cumulated_reward += reward #add current reward to cumulated reward

            #print("Observation/State: \n{}".format(observation_focus))
            #print("Reward: {:+0.2f}".format(reward))
            #print("Done: {}".format(done))
            #print("Info: {}".format(info))

        # STEP (7B.13): #Print results about current iteration
        print("Cumulative reward: {:+0.2f}".format(cumulated_reward))
        print("Average reward per iteration: {:+0.3f}".format(cumulated_reward / duration_of_sim))


        cumulated_reward_added_up += cumulated_reward
        average_reward_added_up += (cumulated_reward / duration_of_sim)
        list_of_average_rewards = np.append(list_of_average_rewards, (cumulated_reward / duration_of_sim))

    env.close()

    print('**********************************')
    print("Testing is done.")
    print("**********************************")
    # STEP (7B.14): #Print overall results
    print("Results after {} iterations:".format(n_iterations_testing))
    print("**********************************")
    #print("Average cumulative reward {:+0.2f}".format(cumulated_reward_added_up / n_iterations_testing))
    #print("Average reward per iteration for all iterations (average) {:+0.3f}".format(average_reward_added_up / n_iterations_testing))

    list_of_average_rewards_rounded = ['%.3f' % elem for elem in list_of_average_rewards] #round list entries
    print("This list of all average rewards is:")
    print(', '.join(list_of_average_rewards_rounded)) #makes the print without quotes

    print("Average reward per iteration for all iterations (average) {:+0.3f}".format(np.mean(list_of_average_rewards)))
    print("Standard deviation: {:+0.3f}".format(np.std(list_of_average_rewards)))

    #inspired by https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/


#######################################################################################################################
#STEP (7C): #DQN Learning (not successfully implemented yet TODO)
#######################################################################################################################
elif (mode == "reinforcement-learning-deep-q-learning"):
        n_iterations_deep_q = 52

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
            state_focus = custom_observation_conversion(state) #only use 9 pixels; and discretize values; and convert to decimal index
            # todo ggf. anpassen, weil hier die Vereinfachung vllt gar nicht erforderlich....
            #state_focus = np.reshape(state_focus, [1, state_space_size_combinations])
            reward = 0
            done = False
            batch_size = 32

            # time_t represents each frame of the game
            # Our goal is to keep the pole upright as long as possible until score of 500
            # the more time_t the more score
            for t in range(duration_of_sim):
                print("Starting Step {0} in Iteration {1}".format(t, i_episode))
                if (render_active): env.render()  # render one frame

                # Decide action
                action = agent.act(state)

                print("Action: {}".format(action))

                next_state, reward, done, info = env.step(action)   # Advance the game to the next frame based on the action.
                next_state_focus = custom_observation_conversion(next_state) #only use 9 pixels; and discretize values; and convert to decimal index

                #next_state_focus = np.reshape(next_state_focus, [1, state_space_size_combinations]) #todo warum nicht von 0 bis n-1 (bei dem anderen auch)

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
