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


class SumTree(object):
    data_pointer = 0
    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        #         # [--------------data frame-------------]
        #         #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1  # + parentnode
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  #
            tree_idx = (tree_idx -1) // 2  # 向下取整
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
          Tree structure and array storage:
          Tree index:
               0         -> storing priority sum
              / \
            1     2
           / \   / \
          3   4 5   6    -> storing priority for transitions
          Array type for storing:
          [0,1,2,3,4,5,6]
          """
        parent_idx = 0
        while True:  #
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p

    def sample(self, n):
        # prob = p / self.tree.total_p
        # ISWeight = (N*Pj)^(-beta) / maxi_wi = = (N*Pj)^(-beta) / maxi[ (N*Pi)^(-beta) ]
        # maxi[ (N*Pi)^(-beta)] = mini[ (N*Pi) ] ^ (-beta) ?
        # ISWeight = (Pj / mini[Pi])^(-beta)
        b_idx = np.empty((n,), dtype=np.int32)
        b_memory = np.empty((n, self.tree.data[0].size))
        ISWeights = np.empty((n, 1))

        pri_seg = self.tree.total_p / n  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight
        if min_prob == 0:
            min_prob = 0.00001
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


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


class DDQN_PER(object):
    def __init__(self, ):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        # self.memory_counter = 0                                         # for storing memory
        # self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.memory = Memory(capacity=MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, observation):
        observation = torch.unsqueeze(torch.FloatTensor(observation), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(observation)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        self.memory.store(transition)  # have high priority for newly arrived transition

        # replace the old memory with new memory
        # index = self.memory_counter % MEMORY_CAPACITY
        # self.memory[index, :] = transition
        # self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        tree_idx, batch_memory, ISWeight = self.memory.sample(BATCH_SIZE)

        # sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        # b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(batch_memory[:, :N_STATES])
        b_a = torch.LongTensor(batch_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(batch_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(batch_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        # ddqn
        q_eval = self.eval_net(b_s)
        q_eval_ = self.eval_net(b_s_)
        q_eval_4next = q_eval_.max(1)[1].view(BATCH_SIZE, 1)  # 返回每一行最大q_eval值的索引
        q_next = self.target_net(b_s_).detach().gather(1, q_eval_4next)

        q_target = b_r + GAMMA * q_next
        # loss = self.loss_func(q_eval.gather(1, b_a), q_target)
        ISWeight = torch.from_numpy(ISWeight)
        loss = torch.mean(ISWeight * torch.pow(q_eval.gather(1, b_a) - q_target, 2))

        abs_errors = torch.abs(q_eval.gather(1, b_a) - q_target).detach().numpy()
        self.memory.batch_update(tree_idx, abs_errors)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

ddqn_per = DDQN_PER()

print('\nCollecting experience...')
reward_record = []
for i_episode in range(300):
    s = env.reset()
    ep_r = 0
    while True:
        env.render()
        a = ddqn_per.choose_action(s)

        # take action
        s_, r, done, info = env.step(a)

        # modify the reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        ddqn_per.store_transition(s, a, r, s_)

        ep_r += r
        ddqn_per.learn()
        if done:
            print('Ep: ', i_episode,
                  '| Ep_r: ', round(ep_r, 2))
            reward_record.append(ep_r)

        if done:
            break
        s = s_
env.close()
reward_record = np.array(reward_record)
np.save('ddqn_per.npy', reward_record)  # 保存为.npy格式
data = np.load('ddqn_per.npy')
data = data.tolist()
plt.plot(data)
plt.show()
