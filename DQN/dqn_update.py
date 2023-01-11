#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
from collections import namedtuple

import random


# In[2]:


# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
# env = gym.make('CartPole-v0')
env = gym.make('CartPole-v1')
# env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


# In[5]:


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


# In[6]:


transition = namedtuple('transition',
                       ('state','action','reward','next_state'))

class ReplayMemory(object):
    def __init__(self,capacity):
        self.capacity = MEMORY_CAPACITY
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition(*args)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


# In[7]:


class DQN(object):
    def __init__(self, ):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.replay_memory = ReplayMemory(MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
#         print("x: ", x)
#         print("torch.FloatTensor(x): ", torch.FloatTensor(x))        
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        transitions = self.replay_memory.sample(BATCH_SIZE)
        batch = transition(*zip(*transitions))
#         print("batch: ", batch)
        b_s = torch.FloatTensor(np.array(batch.state))
#         print("b_s: ", b_s.shape)
        b_a = torch.LongTensor(np.array(batch.action)).unsqueeze(1)
#         print("b_a: ", b_a.shape)
        b_r = torch.FloatTensor(np.array(batch.reward)).unsqueeze(1)
#         print("b_r: ", b_r.shape)
        b_s_ = torch.FloatTensor(np.array(batch.next_state))
#         print("b_s_: ", b_s_.shape)

        # q_eval w.r.t the action in experience
        # dqn
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# In[8]:


dqn = DQN()


# In[9]:


print('\nCollecting experience...')
reward_record = []
for i_episode in range(300):
    s = env.reset()
    s = s[0]
    ep_r = 0
    while True:
#         env.render()
#         print("s: ", s)
        a = dqn.choose_action(s)

        # take action
#         print("env.step(a): ", env.step(a))
        s_, r, terminated,  truncated,info = env.step(a)
    
        done = terminated or truncated

        # modify the reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r_modify = r1 + r2

#         dqn.store_transition(s, a, r_modify, s_)
        dqn.replay_memory.push(s, a, r_modify, s_)

        ep_r += r
        if len(dqn.replay_memory) == MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))
                reward_record.append(ep_r)

        if done:
            break
        s = s_


# In[10]:


env.close()
reward_record = np.array(reward_record)
np.save('dqn.npy', reward_record)  # 保存为.npy格式
data = np.load('dqn.npy')
data = data.tolist()
plt.plot(data)
plt.show()


# In[ ]:




