#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 22:19:48 2019

@author: fbessai
"""
import numpy as np
import gym
from dqn import Dqn, State
from matplotlib import pyplot as plt


NB_EPISODES = 1000  # max number of episodes  
NB_STEPS_PER_EPISODE = 1000 

LEARN_NN_SIZE = [80, 150] # Neural net size by layer
LEARN_EPS_DECAY = .993 #  Exploration rate decay. default =.993 
LEARN_ACTION_DEPTH = 5 # Depth of eligibility trace. 
LEARN_ACTION_DISCOUNT = .99  # discount factor for eligibility trace

LEARN_EPISODES = 700 # number of learning episodes



RESULT_TOTALS = [] # set of the accumulated reward per episode


env = gym.make('LunarLander-v2') # create gym the environment


print(env.action_space.n)
print(env.observation_space.shape[0])

nnet = Dqn(env.action_space, env.observation_space.shape[0], LEARN_NN_SIZE, LEARN_EPS_DECAY, LEARN_ACTION_DEPTH, LEARN_ACTION_DISCOUNT)

        
        

class Action:  # action space 
    DO_NOTHING  = 0 
    FIRE_LEFT   = 1
    FIRE_BOTTOM = 2
    FIRE_RIGHT  = 3
    ACTIONS = np.array([DO_NOTHING,FIRE_LEFT, FIRE_BOTTOM,FIRE_RIGHT])
    
    def getActionFromVec(actionVec):
        return  np.select( actionVec, Action.ACTIONS)
    

TARGET = State([0,0]);



def dqn_policy(state, explore): # determines the action to take given the state
   
    actions = nnet.computeAction(state, explore)
    

    return Action.getActionFromVec(actions)



for episode in range(NB_EPISODES):
    
    print('------- Episode : '+ str(episode))
    episode_rewards = 0 
    state = State(env.reset()) # initial state of the environement 
    action = Action.DO_NOTHING 
    
    learn = episode <= LEARN_EPISODES;
    

    for step in range (NB_STEPS_PER_EPISODE): # max number of steps for one episode
        
        
        action = dqn_policy(state.obs, learn) # compute the action to perform based on the current state of the enviroent 
        
        env.render() # show the scene
            
        next_state, reward, done, info = env.step(action) # perform the action and get the reward and new state
        
        state.action = action
        state.reward = reward
        state.done =  done
        
        state.next_state = State(next_state)
        
        if(learn):
            nnet.addToMemory(state) # memorize the step transition values
        
        
        state = state.next_state
        
        
        episode_rewards += reward
        
        if done:

            RESULT_TOTALS.append(episode_rewards)
            break
    
    if(learn):    
        nnet.learn(np.average(RESULT_TOTALS)) 

    plt.plot(RESULT_TOTALS)
    plt.show()
   
print(RESULT_TOTALS)
print('Max reward : ' + str(max(RESULT_TOTALS)))

plt.plot(RESULT_TOTALS)
plt.show()

env.close()
env.reset()