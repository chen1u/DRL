import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v0')
# env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


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


class DDQN(object):
    def __init__(self, double_q=True):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
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

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        # ddqn
        q_eval = self.eval_net(b_s)
        # print('q_eval: ', q_eval)
        # q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        # print('b_a: ', b_a)
        # c = q_eval.gather(1, b_a)
        # print('c: ', c)
        # q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        # q_eval4next = q_eval.max(1)[1].unsqueeze(0)
        q_eval_ = self.eval_net(b_s_)
        q_eval_4next = q_eval_.max(1)[1].view(BATCH_SIZE, 1)  # 返回每一行最大q_eval值的索引
        # print('q_eval4next: ', q_eval4next)
        # print('q_eval.gather(1, q_eval4next): ', q_eval.gather(1, q_eval4next))
        q_next = self.target_net(b_s_).detach().gather(1, q_eval_4next)
        # q_next = self.target_net(b_s_).detach()
        # print('q_next: ', q_next)
        # d = q_next.gather(1, q_eval4next)
        # print('d: ', d)

        # q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        q_target = b_r + GAMMA * q_next
        # print('q_target: ', q_target)
        loss = self.loss_func(q_eval.gather(1, b_a), q_target)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

ddqn = DDQN()

print('\nCollecting experience...')
reward_record = []
for i_episode in range(300):
    s = env.reset()
    ep_r = 0
    while True:
        env.render()
        a = ddqn.choose_action(s)

        # take action
        s_, r, done, info = env.step(a)

        # modify the reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        ddqn.store_transition(s, a, r, s_)

        ep_r += r
        if ddqn.memory_counter > MEMORY_CAPACITY:
            ddqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))
                reward_record.append(ep_r)

        if done:
            break
        s = s_
env.close()
reward_record = np.array(reward_record)
np.save('ddqn.npy', reward_record)  # 保存为.npy格式
data = np.load('ddqn.npy')
data = data.tolist()
plt.plot(data)
plt.show()
