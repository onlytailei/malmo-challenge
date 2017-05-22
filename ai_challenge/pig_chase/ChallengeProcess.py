#!/usr/bin/env python
# coding=utf-8
'''
Author:Tai Lei
Date:So 21 Mai 2017 12:10:07 CEST
Info:
    '''
import sys
sys.path.append('./pytorch_rl/')

from threading import Thread, active_count
import thread
from core.env import Env
from time import sleep
import multiprocessing as mp

from common import parse_clients_args
from malmopy.agent import RandomAgent
from agent import PigChaseChallengeAgent
from environment import PigChaseEnvironment, PigChaseSymbolicStateBuilder, PigChaseTopDownStateBuilder,PigChaseTopDownStateBuilder4_channel

class ChallengeProcess(mp.Process):
    def __init__(self, process_id=0):
        super(ChallengeProcess, self).__init__(name = "Challenge Process-%d" % process_id)
        
        self.process_id = process_id
    
        print ("Oppenent init") 
        
        self.clients = parse_clients_args(["malmo"+str((process_id+1)*2-1)+':10000', "malmo"+str((process_id+1)*2)+':10000']) 
    
    def run(self):
        
        builder = PigChaseSymbolicStateBuilder()
        self.env = PigChaseEnvironment(self.clients,        
                builder, 
                role= 0,
                randomize_positions=True)
        print ("Oppenent Start") 
        agent = PigChaseChallengeAgent('Agent_1')
        self.env.reset()
        print ("Oppenent env and agent") 
    
        if type(agent.current_agent) == RandomAgent:
            agent_type = PigChaseEnvironment.AGENT_TYPE_1
        else:
            agent_type = PigChaseEnvironment.AGENT_TYPE_2

        obs = self.env.reset(agent_type)
        reward = 0
        agent_done = False
        print ("Oppenent initial_over") 

        while True:
            if self.env.done:
                print ("Oppenent Done") 
                if type(agent.current_agent) == RandomAgent:
                    agent_type = PigChaseEnvironment.AGENT_TYPE_1
                else:
                    agent_type = PigChaseEnvironment.AGENT_TYPE_2
                
                obs = env.reset(agent_type)
                while obs is None:
                    print ('Warning: received obs == None.')
                    obs = self.env.reset(agent_type)

            # select an action
            action = agent.act(obs, reward, agent_done, is_training=True)
            # take a step
            obs, reward, agent_done = self.env.do(action)


