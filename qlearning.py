# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 16:34:29 2025

@author: asus
"""
import gym
import  random
import numpy as np
environment=gym.make("FrozenLake-v1",is_slippery=False,render_mode="ansi")
environment.reset()
nb_states=environment.observation_space.n
nb_actions=environment.action_space.n
qtable=np.zeros((nb_states,nb_actions))
print("qtable")
print(qtable)
action=environment.action_space.sample()
"""
sol->0,
asagi->1,
sag->2,
yukarı->3
"""
new_state,reward,done,info,_=environment.step(action)
#%%

import gym
import  random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
environment=gym.make("FrozenLake-v1",is_slippery=False,render_mode="ansi")
environment.reset()
nb_states=environment.observation_space.n
nb_actions=environment.action_space.n
qtable=np.zeros((nb_states,nb_actions))#ajanin beyni
print("qtable")
print(qtable)
episodes=1000
alpha=0.5#learning rate
gamma=0.9#discount rate
outcomes=[]

#training
for _ in tqdm(range(episodes)):
    state,_=environment.reset()
    done=False#ajanın basari durumu
    
    outcomes.append("Failure")
    while not done:#ajan basasrili olana kadar state icinde hareket et
        if np.max(qtable[state]>0):
            action=np.argmax(qtable[state])
        else:
            action=environment.action_space.sample()
            
        new_state,reward,done,info,_=environment.step(action)
        #update qtable
        qtable[state,action]= qtable[state,action]+alpha*(reward+gamma*np.max(qtable[new_state])-qtable[state,action])  
        state=new_state
        
        if reward:
            outcomes[-1]="Success"
            
print("qtable after training")
print(qtable)

plt.bar(range(episodes),outcomes)
#test
episodes=100
nb_success=0


for _ in tqdm(range(episodes)):
    state,_=environment.reset()
    done=False#ajanın basari durumu
    
    
    while not done:#ajan basasrili olana kadar state icinde hareket et
        if np.max(qtable[state]>0):
            action=np.argmax(qtable[state])
        else:
            action=environment.action_space.sample()
            
        new_state,reward,done,info,_=environment.step(action)
        #update qtable
        qtable[state,action]= qtable[state,action]+alpha*(reward+gamma*np.max(qtable[new_state])-qtable[state,action])  
        state=new_state
        nb_success+=reward
print("success rate:",100*nb_success/episodes)
