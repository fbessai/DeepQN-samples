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




class Dqn:


    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        
        self.lr = 0.001  # learning rate 
        self.discount = .99 # discount factor of the future rewards
        
        self.eps = 1.0 # initial exploration rate. Its value evolves during learning
        self.eps_decay = .993  # the decrease rate of the exploration rate
        self.eps_min = .01 # the minimum exploration rate allowed 
        
        
        self.memory = deque(maxlen=10000) # the states transition memory
        self.batch_size = 100  # the number of transition used for updateing the model 
        self.min_samples_to_learn= 1000 #  minimum number of transition in the memory before to start learning
        
        self.nn = self.build_nnet()
        

    def build_nnet(self):

        nn = Sequential()
        nn.add(Dense(150, activation='relu', input_shape=( self.state_space,)))
        nn.add(Dense(120, activation='relu'))
        nn.add(Dense(self.action_space, activation='linear'))
        nn.compile(loss='mse', optimizer=optimizers.Adam(lr=self.lr))
        return nn
    
    def addToMemory(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done));
    
    def memorySize(self):
        return len(self.memory)
        
    def learn(self):
        
        batch_size =  self.memorySize()
        if(batch_size<self.min_samples_to_learn):
            return
            
        minibatch = random.sample(self.memory,batch_size)
        states, actions, reward, nextStates, dones = zip(*minibatch)
        states = np.asarray(states)
        actions = np.asarray(actions)
        reward = np.asarray(reward)
        nextStates = np.asarray(nextStates)
        dones = np.asarray(dones)
        
        targetReward = reward # by default, which corresponds to DONE
        
        #The value of the rewards should be updated only in case this is not the final step (DONE=False) 
        predActionsRewards = self.nn.predict(nextStates, verbose=0)
        pred2 = np.amax(predActionsRewards,axis=1)
        pred3 = (reward + self.discount * pred2)[np.invert(dones)]
        targetReward[np.invert(dones)] = pred3
        
        output = self.nn.predict(states, verbose=0)
        output[np.arange(0,len(output)), actions] = targetReward
        
        
        self.nn.fit(states, output, epochs=1, verbose=0)
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay

        print('------- Epsilon: '+ str(self.eps))
    def computeAction(self, state):
    
        selected = 0
        if np.random.rand() <= self.eps:
            selected = random.randrange(self.action_space)
        else:
        
            mstate = np.array([state]) #because predict takes on a collection of states
            rawAction = self.nn.predict(mstate, verbose=0)
        

            selected = np.argmax(rawAction[0])
            
        actions = np.zeros(self.action_space, dtype=bool)
        actions[selected]=True
        
        
        
    
        return actions



