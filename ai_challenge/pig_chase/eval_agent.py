#!/usr/bin/env python
# coding=utf-8
'''
Author:Tai Lei
Date:Mo 22 Mai 2017 10:13:22 CEST
Info:
'''
from malmopy.agent import BaseAgent
from model import ActorCritic
from torch.autograd import Variable
import torch
import numpy as np
from environment import PigChaseTopDownStateBuilder4_channel
from evaluation import PigChaseEvaluator

class MyTrainedAgent():
    
    def __init__(self, weight_path):
        
        self.model = ActorCritic(4,3)
        self.model.load_state_dict(torch.load(weight_path))
        self.cx = Variable(torch.zeros(1, 256))
        self.hx = Variable(torch.zeros(1, 256))


    def act(self, state, reward, done, is_training=False):
        
        state_ = state.reshape(1,5,18,18)
        state_ = torch.from_numpy(state_[:1,1:,2:-2,2:-2])
        
        if done:
            self.cx = Variable(torch.zeros(1, 256))
            self.hx = Variable(torch.zeros(1, 256))
            
        value, logit, (self.hx, self.cx) = self.model((Variable(state_),(self.hx,self.cx)))
        action = np.argmax(logit.data.numpy())
        return action


# Creates an agent trained with 100k train calls
my_agent_100k = MyTrainedAgent('/root/malmo-challenge/ai_challenge/pig_chase/weight_100k')

# Creates an agent trained with 500k train calls
my_agent_500k = MyTrainedAgent('/root/malmo-challenge/ai_challenge/pig_chase/weight_500k')

# You can pass a custom StateBuilder for your agent.
# It will be used by the environment to generate state for your agent
eval = PigChaseEvaluator([['malmo1','10000'], ['malmo2','10000']],my_agent_100k, my_agent_500k, PigChaseTopDownStateBuilder4_channel())

# Run and save
eval.run()
eval.save('My experiment 1', '/root/malmo_save/second_result')
