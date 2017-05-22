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

def _opponent_env(clients, pid):
    print ("Oppenent Start") 
    # TODO define clients as process id 
    builder = PigChaseSymbolicStateBuilder()
    
    env = PigChaseEnvironment(clients,        
            builder, 
            role= 0,
            randomize_positions=True)
    
    agent = PigChaseChallengeAgent('Agent_1')
    env.reset()
    print ("Oppenent env and agent") 
    
    if type(agent.current_agent) == RandomAgent:
        agent_type = PigChaseEnvironment.AGENT_TYPE_1
    else:
        agent_type = PigChaseEnvironment.AGENT_TYPE_2

    obs = env.reset(agent_type)
    print (obs)
    reward = 0
    agent_done = False
    print ("Oppenent initial_over") 

    while True:
        if env.done:
            print ("Oppenent Done") 
            if type(agent.current_agent) == RandomAgent:
                agent_type = PigChaseEnvironment.AGENT_TYPE_1
            else:
                agent_type = PigChaseEnvironment.AGENT_TYPE_2
            
            obs = env.reset(agent_type)
            while obs is None:
                print ('Warning: received obs == None.')
                obs = env.reset(agent_type)

        # select an action
        action = agent.act(obs, reward, agent_done, is_training=True)
        # take a step
        obs, reward, agent_done = env.do(action)


class MalmoEnv(Env):
    def __init__(self, args, env_ind=0):
        super(MalmoEnv, self).__init__(args, env_ind)
        assert self.env_type == "malmo"
        
        self.host = args.host
        self.clients = parse_clients_args([self.host+str((env_ind+1)*2-1)+':10000', self.host+str((env_ind+1)*2)+':10000'])
        print (self.clients)
        
        #oppenent agent
        self.oppenent_thread = Thread(target=_opponent_env, kwargs={'clients':self.clients, 'pid':env_ind})
        self.oppenent_thread.daemon=True
        self.oppenent_thread.start()
        sleep(1)
        
        # TODO
        # define clients 
        self.env = PigChaseEnvironment(self.clients, 
                PigChaseTopDownStateBuilder(True),
                role= 1, 
                randomize_positions=True)
        
        # action space setup
        self.actions = range(self.action_dim);
        self.logger.warning("Action Space: %s", self.actions)

        # state space setup
        self.logger.warning("State  Space: %s", self.state_shape)
        print ("init now over")
    
    def __del__(self):
        self.oppenent_thread.terminate()

    @property
    def action_dim(self):
        return self.env.available_actions
    
    
    def _preprocessState(self, state):    # NOTE: here no preprecessing is needed
        return state
    
    @property
    def state_shape(self):
        return (1,1,18,18)

    def render(self):
        pass

    def reset(self):
        print ("Training Thread Done") 
        self._reset_experience()
        self.exp_state1 = self.env.reset()
        while self.exp_state1 is None:
            print ('warning: recieved obs ==None.')
            self.exp_state1 = self.env.reset()
        return self._get_experience()
    
    def step(self, action_index):
        self.exp_action = action_index
        self.exp_state1, self.exp_reward, self.exp_terminal1, _ = self.env.do(self.actions[self.exp_action])
        print ("action_select: ", action_index)
        return self._get_experience()
    
