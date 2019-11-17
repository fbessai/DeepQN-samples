#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 22:19:48 2019

@author: fbessai
"""
import numpy as np
import gym
from dqn import Dqn
from matplotlib import pyplot as plt




env = gym.make('LunarLander-v2') # create gym the environment


print(env.action_space.n)
print(env.observation_space.shape[0])

nnet = Dqn(env.action_space.n, env.observation_space.shape[0])

class State:

    def __init__(self, obs):
        self.coordinates = {}
        self.coordinates['x']= obs[0]
        self.coordinates['y']= obs[1]

class Action:  # action space 
    DO_NOTHING  = 0 
    FIRE_LEFT   = 1
    FIRE_BOTTOM = 2
    FIRE_RIGHT  = 3
    ACTIONS = np.array([DO_NOTHING,FIRE_LEFT, FIRE_BOTTOM,FIRE_RIGHT])
    
    def getActionFromVec(actionVec):
        return  np.select( actionVec, Action.ACTIONS)
    

TARGET = State([0,0]);



def dqn_policy(state): # determines the action to take given the state
   
    actions = nnet.computeAction(state)
    

    return Action.getActionFromVec(actions)

totals = [] # set of the accumulated reward per episode

for episode in range(1000):
    
    print('------- Episode : '+ str(episode))
    episode_rewards = 0 
    state = env.reset() # initial state of the environement 
    action = Action.DO_NOTHING 

    for step in range (1000): # max number of steps for one episode
        
        
        action = dqn_policy(state) # compute the action to perform based on the current state of the enviroent 
        
        env.render() # show the scene
            
        next_state, reward, done, info = env.step(action) # perform the action and get the reward and new state
        
        if(episode <=400):
            nnet.addToMemory(state, action, reward, next_state, done) # memorize the step transition values
        
        
        episode_rewards += reward
        
        state = next_state
        
        if done:

            totals.append(episode_rewards)
            break
    
    if(episode <=400):    
        nnet.learn() 

    plt.plot(totals)
    plt.show()
   
print(totals)
print('Max reward : ' + str(max(totals)))

plt.plot(totals)
plt.show()

env.close()
env.reset()