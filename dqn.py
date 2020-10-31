#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 18:06:40 2019

@author: fbessai
"""
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import optimizers
from keras.utils import np_utils
from collections import deque


class State:

    def __init__(self, obs):
        self.obs= obs
        self.next_state = None
        self.action = None
        self.reward = 0.0
        self.done = False
        
        
    def q_value(self, nn, discount, n_steps):
        if( self.next_state == None  or self.done):
            return self.reward
        
        
        future_reward = 0.0
        if n_steps<=1 :
            q_values = nn.predict(np.array([self.next_state.obs]), verbose=0)
            future_reward = np.amax(q_values,axis=1)[0]
        else:
            future_reward = self.next_state.q_value(nn, discount,n_steps-1)
        
        return self.reward + discount * future_reward
    
    def next_state(self):
        return self.next_state
    
    def set_next_state(self, state):
        self.next_state = state
    
class Dqn:


    def __init__(self, action_space, state_size, layer_sizes, learn_eps_decay, learn_action_depth, learn_action_discount):

        self.action_space = action_space
        self.state_size = state_size
        self.layer_sizes = layer_sizes
        
        self.learn_action_depth =  learn_action_depth
        
        self.lr = 0.001  # learning rate 
        self.discount = learn_action_discount # discount factor of the future rewards
        
        self.eps = 1.0 # initial exploration rate. Its value evolves during learning
        self.eps_decay = learn_eps_decay  # the decrease rate of the exploration rate
        self.eps_min = .01 # the minimum exploration rate allowed 
        
        
        self.memory = deque(maxlen=40000) # the states transition memory
        self.batch_size = 3000  # the number of transition used for updateing the model 
        self.min_samples_to_learn= 3000 #  minimum number of transition in the memory before to start learning
        
        self.nn = self.build_nnet()
        

    def build_nnet(self):

        nn = Sequential()
        nn.add(Dense(self.layer_sizes[0], activation='relu', input_shape=( self.state_size,)))
        
        for layer_size in self.layer_sizes:
            nn.add(Dense(layer_size, activation='relu'))
            
        nn.add(Dense(self.action_space.n, activation='linear'))
        nn.compile(loss='mse', optimizer=optimizers.Adam(lr=self.lr))
        return nn
    
    def addToMemory(self, state):
        self.memory.append(state);
    
    def memorySize(self):
        return len(self.memory)
    
    #    avgReward:  the averagge reward so far. used to adjust the explorartion factor
    def learn(self, avgReward):
        

        if(self.memorySize()<self.min_samples_to_learn):
            return
        
  
        states = random.sample(self.memory,self.batch_size)
        
        
        obs = [state.obs for state in states]
        obs = np.asarray(obs)
        
        actions = [state.action for state in states]
        actions = np.asarray(actions)
        
        reward = [state.reward for state in states]
        reward = np.asarray(reward)

        dones = [state.done for state in states]
        dones = np.asarray(dones)
        
        targetReward = reward # by default, which corresponds to DONE
        
        #The value of the rewards should be updated only in case this is not the final step (DONE=False) 
#        predActionsRewards = self.nn.predict(next_obs, verbose=0)
#        
#        pred2 = np.amax(predActionsRewards,axis=1)
#        pred3 = (reward + self.discount * pred2)[np.invert(dones)]
#        targetReward[np.invert(dones)] = pred3
        
        
        targetReward = [state.q_value(self.nn, self.discount, self.learn_action_depth) for state in states]
        
        output = self.nn.predict(obs, verbose=0)

        output[np.arange(0,len(output)), actions] = targetReward
        
        
        
        self.nn.fit(obs, output, epochs=1, verbose=0)
        
        

        if self.eps > self.eps_min:
            eps = (self.eps * self.eps_decay)  if (avgReward > 0 or self.eps >.20) else (self.eps / self.eps_decay)
            if eps <= 1.0:
                self.eps = eps

        print('------- Epsilon: '+ str(self.eps))
        
        
    def computeAction(self, obs, explore):
    
        selected = 0
        
        if explore and np.random.rand() <= self.eps:
            # exploration
            selected = self.action_space.sample()
        else:
        
            mstate = np.array([obs]) #because predict takes on a collection of states
            rawAction = self.nn.predict(mstate, verbose=0)

        

            selected = np.argmax(rawAction[0])  # get the index of the item having the max value
            
        actions = np.zeros(self.action_space.n, dtype=bool)
        actions[selected]=True
        
        
        
    
        return actions



