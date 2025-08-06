# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 17:04:27 2025

@author: asus
"""

import gym
import  random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
env=gym.make("Taxi-v3",render_mode="ansi")
env.reset()
print(env.render())
"""
0:guney
1:kuzey
2:dogu
3:bati
4:yolcuyu al
5:yolcuyu bırak"""
action_space=env.action_space.n
state_spece=env.observation_space.n
q_table=np.zeros((state_spece,action_space))
alpha=0.1
gamma=0.6
epsilon=0.1
for i in tqdm(range(1,100001)):
    state,_=env.reset()
    done=False
    while not done:
        if random.uniform(0,1)<epsilon:#explore %10
           action=env.action_space.sample() 
            
        else:#exploit bildigini yapmak
            action=np.argmax(q_table[state])
            
        next_state,reward,done,info,_=env.step(action)
        q_table[state,action]= q_table[state,action]+alpha*(reward+gamma*np.max(q_table[next_state])-q_table[state,action])
        state=next_state
print("training finished")  
#test
total_epoch,total_penalties=0,0
episodes=100
for i in tqdm(range(episodes)):
    state,_=env.reset()
    epochs,penalties,reward=0,0,0
    done=False
    while not done:    
        action=np.argmax(q_table[state])
            
        next_state,reward,done,info,_=env.step(action)
        state=next_state
        if reward==-10:
            penalties+=1
            
        epochs=+1
    total_epoch+=epochs
    total_penalties+=penalties
    print("REsult after{}episodes".format(episodes))
    print("Avarage timesteps per episode:",total_epoch/episodes)
    print( "Avarage penalties per episodes: ",total_penalties/episodes)
  
              






