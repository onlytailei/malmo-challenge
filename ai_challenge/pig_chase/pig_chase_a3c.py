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
import multiprocessing as mp
import numpy as np
import scipy.ndimage

from malmopy.agent import LinearEpsilonGreedyExplorer, RandomAgent
from malmopy.model.chainer import QNeuralNetwork, DQNChain

from common import parse_clients_args, visualize_training, ENV_AGENT_NAMES
from agent import PigChaseChallengeAgent, PigChaseQLearnerAgent
from environment import PigChaseEnvironment, PigChaseSymbolicStateBuilder, \
    PigChaseTopDownStateBuilder

from malmopy.agent import TemporalMemory

from model import ActorCritic
import my_optim
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from multiprocessing import Value
import visdom
#try:
    #from malmopy.visualization.tensorboard import TensorboardVisualizer
    #from malmopy.visualization.tensorboard.cntk import CntkConverter
#except ImportError:
    #print('Cannot import tensorboard, using ConsoleVisualizer.')
    #from malmopy.visualization import ConsoleVisualizer

# Torch Area


# Enforce path
sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.join(os.path.pardir, os.getcwd()))

DQN_FOLDER = 'results/baselines/%s/dqn/%s-%s'
EPOCH_SIZE = 100000


def agent_factory(name, role, clients, device, max_epochs, logdir, shared_model, optimizer, args, main_step, vis=None):
    #assert len(clients) >= 2, 'Not enough clients (need at least 2)'
    clients = parse_clients_args(clients)

    if role == 0:

        builder = PigChaseSymbolicStateBuilder()
        env = PigChaseEnvironment(clients, builder, role=role,
                                  randomize_positions=True)

        agent = PigChaseChallengeAgent(name)
        if type(agent.current_agent) == RandomAgent:
            agent_type = PigChaseEnvironment.AGENT_TYPE_1
        else:
            agent_type = PigChaseEnvironment.AGENT_TYPE_2

        obs = env.reset(agent_type)
        reward = 0
        agent_done = False

        while True:
            if env.done:
                if type(agent.current_agent) == RandomAgent:
                    agent_type = PigChaseEnvironment.AGENT_TYPE_1
                else:
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
    else:
        env = PigChaseEnvironment(clients, PigChaseTopDownStateBuilder(True),
                                  role=role, randomize_positions=True)
        
        #memory = TemporalMemory(100000, (18, 18))
        #chain = DQNChain((memory.history_length, 18, 18), env.available_actions)
        #target_chain = DQNChain((memory.history_length, 18, 18), env.available_actions)
        #model = QNeuralNetwork(chain, target_chain, device)
        #explorer = LinearEpsilonGreedyExplorer(1, 0.1, 1000000)
        #agent = PigChaseQLearnerAgent(name, env.available_actions,
                                      #model, memory, 0.99, 32, 50000,
                                      #explorer=explorer, visualizer=visualizer)
        
        model = ActorCritic(1,3) # env state channel and action space
        model.train() 
         
        state_ = env.reset()
        state_ = state_.reshape(1,1,18,18)
        #print state.shape
        state = torch.from_numpy(state_)
        done = True
        
        max_training_steps = EPOCH_SIZE * max_epochs
        episode_length = 0
        loss_history = []
        win = None
        img_win = None
        while True:
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
                state_ = state_.reshape(1,1,18,18)
                done = done or episode_length >= args.max_episode_length
                #reward = max(min(reward, 1), -1)

                if done:
                    episode_length = 0
                    state_ = env.reset()
                    while state_ is None:
                        # this can happen if the episode 
                        # ended with the first
                        # action of the other agent
                        print('Warning: received obs==None.')
                        state_ = env.reset()
                    state_ = state_.reshape(1,1,18,18)

                state = torch.from_numpy(state_)
                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)
                
                #print state.numpy().shape, done, reward

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


def loss_visual(vis, loss_history, main_step, win):
    if main_step.value > 3:
        Y_ = np.array(loss_history).reshape(-1,1)
        win = vis.line(Y = Y_, X = np.arange(len(loss_history)), win=win)
        return win 

def image_visual(vis, img, win):
    img_ = np.reshape(img, (18,18))
    img_ = scipy.ndimage.zoom(img_, 10, order=0)
    img_ = np.stack((img_,)*3)
    #img_f = scipy.ndimage.zoom(img_, 10, order=0)
    #print img_f.shape
    #img_ = cv2.resize(img_,(160,160))
    win = vis.image(img_, win=win)
    return win 

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def test_process(model, step):
    if step.value%5000==0:
        torch.save(model.state_dict(), './models/'+str(step.value)+'_weight')
    sleep(10)

def run_experiment(agents_def):
    #assert len(agents_def) == 2, 'Not enough agents (required: 2, got: %d)'% len(agents_def)
    shared_model = ActorCritic(1,3)
    shared_model.share_memory()
    optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=0.0001)
    optimizer.share_memory()
    main_step = Value('l',0) 
    vis = visdom.Visdom()
    
    processes = []
    test_p = mp.Process(target=test_process, kwargs={'model':shared_model, 'step':main_step})
    test_p.start()
    processes.append(test_p)
        
    for agent in agents_def:
        agent['optimizer'] = optimizer
        agent['shared_model'] = shared_model
        agent['vis'] = vis
        agent['main_step'] = main_step
        #p = Thread(target=agent_factory, kwargs=agent)
        p = mp.Process(target=agent_factory, kwargs=agent)
        #p.daemon = True
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
    arg_parser = ArgumentParser('Pig Chase DQN experiment')
    arg_parser.add_argument('-e', '--epochs', type=int, default=5,
                            help='Number of epochs to run.')
    arg_parser.add_argument('clients', nargs='*',
            default=['127.0.0.1:10000', '127.0.0.1:10001', '127.0.0.1:10002', '127.0.0.1:10003'],
                            help='Minecraft clients endpoints (ip(:port)?)+')
    arg_parser.add_argument('-d', '--device', type=int, default=-1,
                            help='GPU device on which to run the experiment.')
    arg_parser.add_argument('-n', '--number_paralle', type=int, default=2,
                            help='number of a3c processes.')
    arg_parser.add_argument('--num-steps', type=int, default=10, metavar='NS',
                    help='number of forward steps in A3C (default: 20)')
    arg_parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
    arg_parser.add_argument('--max-episode-length', type=int, default=10000, 
            metavar='M',help='maximum length of an episode (default: 10000)')
    arg_parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
    arg_parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 4)')
    args = arg_parser.parse_args()
    
    logdir = path.join('results/pig_chase/dqn', datetime.utcnow().isoformat())
    
    agents = []
    for item in xrange(args.num_processes): 
        agents += [{'name': agent, 'role': role, 'clients': args.clients[item*2:item*2+2],
                   'device': args.device, 'max_epochs': args.epochs,
                   'logdir': logdir, 'args':args}
                  for role, agent in enumerate(ENV_AGENT_NAMES)]
    run_experiment(agents)
