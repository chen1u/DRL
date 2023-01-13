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

import random, math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Hyper Parameters
BATCH_SIZE = 32
# LR = 0.01                   # learning rate
LR = 0.00025
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 4   # target update frequency
MEMORY_CAPACITY = 50000
env = gym.make('CartPole-v1')
# env = gym.make('MountainCar-v0')
# env = gym.make('Acrobot-v1')
# env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


# In[3]:


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)


# In[4]:


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(N_STATES, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, N_ACTIONS)
        )

    def forward(self, x):
        return self.layers(x)


# In[5]:


# epslion_greedy
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


# In[6]:


plt.plot([epsilon_by_frame(i) for i in range(100000)])


# In[7]:


transition = namedtuple('transition',
                       ('state','action','reward','next_state','done'))

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


# In[8]:


class DQN(object):
    def __init__(self, ):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.replay_memory = ReplayMemory(MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
#         self.optimizer = torch.optim.Adam(self.eval_net.parameters())    
        self.loss_func = nn.MSELoss()

    def choose_action(self, x, epsilon):   
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if random.random() > epsilon:   # greedy
            actions_value = self.eval_net.forward(x)
#             print("actions_value: ", actions_value)
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
        b_s = torch.FloatTensor(np.array(batch.state))
        b_a = torch.LongTensor(np.array(batch.action))
        b_r = torch.FloatTensor(np.array(batch.reward))
        b_s_ = torch.FloatTensor(np.array(batch.next_state))
        b_d = torch.FloatTensor(np.array(batch.done))
#         print("b_s: ", b_s)
#         print("b_a: ", b_a)
#         print("b_r: ", b_r)
#         print("b_s_: ", b_s_)
#         print("b_d: ", b_d)
        
        # compute TD loss
        q_values = self.eval_net(b_s)
        next_q_values = self.target_net(b_s_)
#         next_q_values = self.eval_net(b_s_)
#         print("q_values: ", q_values)
#         print("next_q_values: ", next_q_values)
        
        q_value = q_values.gather(1, b_a.unsqueeze(1)).squeeze(1)
#         print("q_value: ", q_value)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = b_r + GAMMA * next_q_value * (1 - b_d)
#         expected_q_value = b_r + GAMMA * next_q_value
        
#         q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
#         q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
#         q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


# In[9]:


dqn = DQN()


# In[10]:


print('\nCollecting experience...')
reward_record = []
losses = []
i_episode = 0
num_frames = 50000

s = env.reset()[0]
ep_r = 0

for frame_idx in range(1, num_frames + 1):
    epsilon = epsilon_by_frame(frame_idx)
#     print("epsilon: ", epsilon)
    a = dqn.choose_action(s, epsilon)
#     print("a: ", a)

    # take action
    s_, r, terminated,  truncated,info = env.step(a)

    done = terminated or truncated

    # no modify
    r_modify = r

    dqn.replay_memory.push(s, a, r_modify, s_, done)
    s = s_

    ep_r += r
    if len(dqn.replay_memory) > BATCH_SIZE:
        loss = dqn.learn()
        losses.append(loss.detach().numpy())
    if done:
        print('Ep: ', i_episode,
              '| Ep_r: ', round(ep_r, 2))
        reward_record.append(ep_r)
        i_episode += 1
        s = env.reset()[0]
        ep_r = 0


# In[11]:


env.close()
reward_record = np.array(reward_record)
np.save('dqn.npy', reward_record)  # 保存为.npy格式
data = np.load('dqn.npy')
data = data.tolist()
plt.plot(data)
plt.show()

losses = np.array(losses)
np.save('losses.npy', losses)
data = np.load('losses.npy')
data = data.tolist()
plt.plot(data)
plt.show()


# In[ ]:




