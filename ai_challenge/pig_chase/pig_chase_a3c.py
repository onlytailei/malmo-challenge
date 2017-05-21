# Copyright (c) 2017 Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===================================================================================================================

import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from os import path
from threading import Thread, active_count
from time import sleep
import time
import multiprocessing as mp
import numpy as np
import scipy.ndimage

from malmopy.agent import RandomAgent
from common import parse_clients_args, ENV_AGENT_NAMES
from agent import PigChaseChallengeAgent
from environment import PigChaseEnvironment, PigChaseSymbolicStateBuilder, PigChaseTopDownStateBuilder,PigChaseTopDownStateBuilder4_channel

from model import ActorCritic
import my_optim
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from multiprocessing import Value
import visdom
import logging

# Enforce path
sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.join(os.path.pardir, os.getcwd()))

EPOCH_SIZE = 100000

def loggerConfig():
    #ts = str(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logger = logging.getLogger()
    formatter = logging.Formatter(
            '%(asctime)s %(levelname)-2s %(message)s')
    logdir = path.join('results/a3c/', datetime.utcnow().isoformat())
    #fileHandler_ = logging.FileHandler(logdir)
    #fileHandler_.setFormatter(formatter)
    #logger.addHandler(fileHandler_)
    logger.setLevel(logging.INFO)
    return logger

def agent_factory(name, role, clients, logger, shared_model, optimizer, args, main_step, vis=None):
    assert len(clients) >= 2, 'Not enough clients (need at least 2)'
    clients = parse_clients_args(clients)
    logger.info("clients: %s, %s", clients[0], clients[1])
    # parterner 
    if role == 0:
        while True:
            builder = PigChaseSymbolicStateBuilder()
            env = PigChaseEnvironment(clients, builder, role=role,
                                      randomize_positions=True)

            agent = PigChaseChallengeAgent(name)
            #if type(agent.current_agent) == RandomAgent:
            #    agent_type = PigChaseEnvironment.AGENT_TYPE_1
            #else:
            agent_type = PigChaseEnvironment.AGENT_TYPE_2

            obs = env.reset(agent_type)
            reward = 0
            agent_done = False

            while True:
                try:
                    if env.done:
                        #if type(agent.current_agent) == RandomAgent:
                        #    agent_type = PigChaseEnvironment.AGENT_TYPE_1
                        #else:
                        agent_type = PigChaseEnvironment.AGENT_TYPE_2
                        obs = env.reset(agent_type)
                        while obs is None:
                            # this can happen if the episode ended with the first
                            # action of the other agent
                            print('Warning: received obs == None.')
                            obs = env.reset(agent_type)

                    # select an action
                    action = agent.act(obs, reward, agent_done, is_training=True)
                    # take a step
                    obs, reward, agent_done = env.do(action)
                except Exception:
                    break
    else:
        while True:
            env = PigChaseEnvironment(clients, 
                    PigChaseTopDownStateBuilder4_channel(True),
                    role=role, 
                    randomize_positions=True)
            
            model = ActorCritic(4,3) # env state channel and action space
            model.train() 
             
            state_ = env.reset()
            while state_ is None:
                # this can happen if the episode 
                # ended with the first
                # action of the other agent
                logger.info('Warning: received obs==None.')
                state_ = env.reset()
            #logger.info("state, %s", state_[:,:10,:10]) 
            state_ = state_.reshape(1,5,18,18)
            state = torch.from_numpy(state_[:1,1:,2:-2,2:-2])
            done = True
            
            #max_training_steps = EPOCH_SIZE
            episode_length = 0
            loss_history = []
            win = None
            img_win = None
            while True:
                try:
                    episode_length += 1
                    model.load_state_dict(shared_model.state_dict())
                     
                    if done:
                        cx = Variable(torch.zeros(1, 256))
                        hx = Variable(torch.zeros(1, 256))
                    else: 
                        cx = Variable(cx.data)
                        hx = Variable(hx.data)
                    
                    values = []
                    log_probs = []
                    rewards = []
                    entropies = []
                        
                    for step in range(args.num_steps):
                        if vis != None:
                            img_win = image_visual(vis, state_, win=img_win)
                        value, logit, (hx, cx) = model(
                            (Variable(state), (hx, cx)))
                        prob = F.softmax(logit)
                        log_prob = F.log_softmax(logit)
                        entropy = -(log_prob * prob).sum(1)
                        entropies.append(entropy)

                        action = prob.multinomial().data
                        log_prob = log_prob.gather(1, Variable(action))
                        state_, reward, done = env.do(action.numpy()[0][0])
                        #logger.info("state: %s", state_)
                        #state_ = state_.reshape(1,5,18,18)
                        #done = done or episode_length >= args.max_episode_length
                        #reward = max(min(reward, 1), -1)

                        if done:
                            episode_length = 0
                            state_ = env.reset()
                            while state_ is None:
                                # this can happen if the episode 
                                # ended with the first
                                # action of the other agent
                                logger.info('Warning: received obs==None.')
                                state_ = env.reset()

                        state_ = state_.reshape(1,5,18,18)
                        state = torch.from_numpy(state_[:1,1:,2:-2,2:-2])
                        values.append(value)
                        log_probs.append(log_prob)
                        rewards.append(reward)

                        if done:
                            break
                
                    R = torch.zeros(1, 1)
                    if not done:
                        value, _, _ = model((Variable(state), (hx, cx)))
                        R = value.data

                    values.append(Variable(R))
                    policy_loss = 0
                    value_loss = 0
                    R = Variable(R)
                    gae = torch.zeros(1, 1)
                    for i in reversed(range(len(rewards))):
                        R = args.gamma * R + rewards[i]
                        advantage = R - values[i]
                        value_loss = value_loss + 0.5 * advantage.pow(2)

                        # Generalized Advantage Estimataion
                        delta_t = rewards[i] + args.gamma * \
                            values[i + 1].data - values[i].data
                        gae = gae * args.gamma + delta_t

                        policy_loss = policy_loss - \
                            log_probs[i] * Variable(gae) - 0.01 * entropies[i]

                    optimizer.zero_grad()

                    loss = policy_loss + 0.5 * value_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(model.parameters(), 40)
                    loss_history.append(loss.data.numpy())
                    if vis != None:
                        win = loss_visual(vis, loss_history, main_step, win)
                    ensure_shared_grads(model, shared_model)
                    optimizer.step()
                    main_step.value +=1
                except Exception:
                    break
def loss_visual(vis, loss_history, main_step, win):
    if main_step.value > 200:
        Y_ = np.array(loss_history).reshape(-1,1)
        win = vis.line(Y = Y_, X = np.arange(len(loss_history)), win=win)
        return win 

def image_visual(vis, img, win):
    img_ = np.reshape(img[0,0,4:-4,4:-4], (10,10))
    img_ = scipy.ndimage.zoom(img_, 10, order=0)
    img_ = np.stack((img_,)*3)
    win = vis.image(img_, win=win)
    return win 

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def test_process(model, step, logger):
    start_time = time.time()
    while True:
        #logger.info("test step value: %d", step.value)
        #if step.value%500==0:
        #    torch.save(model.state_dict(), '/root/malmo_save/'+str(step.value)+'_weight')
        #    logger.info("save model in step %d", step.value)
        #    sleep(10)
        if int(time.time()-start_time)%(17*60)==0:
            torch.save(model.state_dict(), '/root/malmo_save/'+str(step.value)+'_weight')
            torch.save(model.state_dict(), '/root/malmo_save/newest_weight')
       #     logger.info("save model in step %d", step.value)
            sleep(10)

def run_experiment(agents_def):
    assert len(agents_def) >= 2, 'Not enough agents (required: 2, got: %d)'% len(agents_def)
    shared_model = ActorCritic(4,3)
    shared_model.share_memory()
    optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=0.0001)
    optimizer.share_memory()
    main_step = Value('l',0) 
    vis = visdom.Visdom()
    
    if path.isfile('/root/malmo_save/newest_weight'):
        shared_model.load_state_dict(torch.load('/root/malmo_save/newest_weight'))
    
    processes = []
    test_p = mp.Process(target=test_process, kwargs={'model':shared_model, 'step':main_step, 'logger':agents_def[0]['logger']})
    test_p.start()
    processes.append(test_p)
    for agent in agents_def:
        agent['optimizer'] = optimizer
        agent['shared_model'] = shared_model
        #agent['vis'] = None
        agent['vis'] = vis
        agent['main_step'] = main_step
        
        p = mp.Process(target=agent_factory, kwargs=agent)
        p.start()

        # Give the server time to start
        if agent['role'] == 0:
            sleep(1)

        processes.append(p)

    try:
        # wait until only the challenge agent is left
        while active_count() > 2:
            sleep(0.1)
    except KeyboardInterrupt:
        print('Caught control-c - shutting down.')



if __name__ == '__main__':
    arg_parser = ArgumentParser('Pig Chase A3C experiment')
    arg_parser.add_argument('clients', nargs='*',
            default=['10.5.167.15:10000', '10.5.167.15:10001'],
                            help='Minecraft clients endpoints (ip(:port)?)+')
    arg_parser.add_argument('--num-steps', type=int, default=25, metavar='NS',
                    help='number of forward steps in A3C (default: 25)')
    arg_parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
    arg_parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
    arg_parser.add_argument('--num-processes', type=int, default=8, metavar='N',
                    help='how many training processes to use (default: 4)')
    args = arg_parser.parse_args()
    
    logger = loggerConfig()

    agents = []
    for item in xrange(args.num_processes): 
        agents += [{'name': agent, 
                'role': role, 
                'clients': args.clients[item*2:item*2+2], 
                'logger': logger, 
                'args':args}
                for role, agent in enumerate(ENV_AGENT_NAMES)]
    run_experiment(agents)
