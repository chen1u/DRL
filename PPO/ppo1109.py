#!/usr/bin/env python
# coding: utf-8

# In[8]:


get_ipython().run_line_magic('load_ext', 'nb_black')


# In[9]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
from torch.optim import Adam


# In[10]:


# env = gym.make("Pendulum-v1", render_mode="human")
env = gym.make("Pendulum-v1")
# env = env.unwrapped
N_ACTIONS = env.action_space.shape[0]
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = (
    0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape
)  # to confirm the shape
BOUND = env.action_space.high[0]


# In[11]:


# class CriticNet(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(CriticNet, self).__init__()
#         self.layer1 = nn.Linear(in_dim, 64)
#         self.layer2 = nn.Linear(64, 32)
#         self.layer3 = nn.Linear(32, out_dim)

#     def forward(self, obs):
#         if isinstance(obs, np.ndarray):
#             obs = torch.tensor(obs, dtype=torch.float)
#         activation1 = F.relu(self.layer1(obs))
#         activation2 = F.relu(self.layer2(activation1))
#         output = self.layer3(activation2)
#         return output


class CriticNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CriticNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.layer = nn.Sequential(
            nn.Linear(self.in_dim, 128), nn.ReLU(), nn.Linear(128, self.out_dim)
        )

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        v = self.layer(obs)
        return v


# In[12]:


class ActorNet(nn.Module):
    def __init__(self, n_states, bound):
        super(ActorNet, self).__init__()
        self.n_states = n_states
        self.bound = bound

        self.layer = nn.Sequential(nn.Linear(self.n_states, 128), nn.ReLU())

        self.mu_out = nn.Linear(128, 1)
        self.sigma_out = nn.Linear(128, 1)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        obs = F.relu(self.layer(obs))
        mu = self.bound * torch.tanh(self.mu_out(obs))
        sigma = F.softplus(self.sigma_out(obs))
        return mu, sigma


# In[91]:


class PPO:
    def __init__(self):
        self._init_hyperparameters()

        # 更新
        self.actor = ActorNet(N_STATES, BOUND)
        # 采样
        self.actor_sample = ActorNet(N_STATES, BOUND)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)

        self.critic = CriticNet(N_STATES, 1)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # create our variable for the matrix
        # chose 0.5 for stdev arbitrarily
        #         self.cov_var = torch.full(size=(N_ACTIONS,), fill_value=1.0)

        # create the covariance matrix
        #         self.cov_var = torch.diag(self.cov_var)

        self.reward_record = []
        self.V_record = []
        self.v_target_record = []

        # entropy_coef
        self.alpha = 0.01

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 256
        self.max_timesteps_per_episode = 1600
        self.gamma = 0.99
        self.n_updates_per_iteration = 10
        self.clip = 0.2
        self.lr = 3e-4
        self.i_episode = 0

    def get_action(self, obs):
        # for a mean action
        # same thing as calling self.action.forward(obs)
        #         obs = torch.tensor(obs)
        #         print('obs: ', obs)
        mean, sigma = self.actor(obs)

        # create Multivariate Normal Distribution
        #         dist = MultivariateNormal(mean, self.cov_var)
        #         dist = MultivariateNormal(mean, sigma)
        dist = torch.distributions.Normal(mean, sigma)

        # add dist_entropy
        dist_entropy = dist.entropy()

        # sample an action from the distribution and get its log prob
        action = dist.sample()
        # 计算action的log概率
        log_prob = dist.log_prob(action)

        # return the sample action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.
        return (
            np.clip(action.detach().numpy(), -BOUND, BOUND),
            log_prob.detach().numpy(),
            dist_entropy,
        )

    def compute_rtgs(self, batch_r, batch_obs_):
        # the rewards-to-go (rtg) per episode per batch to return
        # the shape will be (num timesteps per episode)
        batch_r_to_go = []
        for ep_r, obs_ in zip(reversed(batch_r), batch_obs_):
            #             discounted_r = self.evaluate(obs_)
            discounted_r = self.evaluate(obs_).detach()
            for r in reversed(ep_r):
                discounted_r = r + discounted_r * self.gamma
                batch_r_to_go.insert(0, discounted_r)

        batch_r_to_go = torch.tensor(batch_r_to_go, dtype=torch.float)
        #         print("type: batch_r_to_go", batch_r_to_go.shape)
        return batch_r_to_go

    #     def compute_rtgs(self, ep_r, ep_obs_):
    #         # the rewards-to-go (rtg) per episode per batch to return
    #         # the shape will be (num timesteps per episode)
    #         batch_r_to_go = []
    #         discounted_r = evaluate(ep_obs_)
    #         for r in reversed(ep_r):
    #             discounted_r = r + discounted_r * self.gamma
    #             batch_r_to_go.insert(0, discounted_r)

    #         batch_r_to_go = torch.tensor(batch_r_to_go, dtype=torch.float)
    #         #         print("type: batch_r_to_go", batch_r_to_go.shape)
    #         return batch_r_to_go

    def rollout(self):
        # batch data
        batch_obs = []
        batch_obs_ = []
        batch_acts = []
        batch_log_probs = []
        batch_r = []
        batch_r_to_go = []
        batch_lens = []  # episodic lengths in batch
        batch_entropy = []

        # Number of timesteps run so far this batch
        t = 0

        while t < self.timesteps_per_batch:
            # rewards of this episode
            ep_r = []
            ep_r_ = []
            obs = env.reset()[0]
            # obs = torch.tensor(env.reset()[0])
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                #                 env.render()
                t += 1
                batch_obs.append(obs)

                action, log_prob, dist_entropy = self.get_action(torch.tensor(obs))
                obs_, r, _, done, _ = env.step(action)
                #                 obs = torch.tensor(obs)

                ep_r.append((r + 8) / 8)
                ep_r_.append(r)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                obs = obs_

                batch_entropy.append(dist_entropy)

                if done:
                    self.i_episode += 1
                    ep_r_sum = sum(ep_r_)
                    if self.i_episode % 10 == 0:
                        print("Ep: ", self.i_episode, "| Ep_r: ", round(ep_r_sum))
                    self.reward_record.append(ep_r_sum)
                    break

            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1)  # plus 1 because timestep starts at 0
            batch_r.append(ep_r)
            batch_obs_.append(obs_)
            #             print("batch_obs_: ", batch_obs_)

            batch_acts = np.array(batch_acts)
            batch_log_probs = np.array(batch_log_probs)

            batch_obs = torch.tensor(batch_obs, dtype=torch.float)
            batch_acts = torch.tensor(batch_acts, dtype=torch.float)
            batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
            #             batch_obs_ = torch.tensor(batch_obs_, dtype=torch.float)

            batch_r_to_go = self.compute_rtgs(batch_r, batch_obs_)
            #             batch_r_to_go.append(self.compute_rtgs(ep_r, ep_obs_))

            #             print("batch_r_to_go: ", len(batch_r_to_go))
            #             print("batch_obs: ", len(batch_obs))

            return (
                batch_obs,
                batch_acts,
                batch_log_probs,
                batch_r,
                batch_r_to_go,
                batch_lens,
                batch_entropy,
            )

    def evaluate(self, batch_obs):
        # query critic network for a value V for each obs in batch_obs
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most
        # recent actor network.
        # This segment of code is similar to that in get_action()
        #         mean = self.actor(batch_obs)
        #         dist = MultivariateNormal(mean, self.cov_var)
        # calculate Ratio of action probabilities
        # with parameters Θ over action probabilities with parameters Θₖ
        #         log_probs = dist.log_prob(batch_acts)

        #         return V, log_probs
        return V

    def learn(self, total_timesteps):
        test = 0
        t_so_far = 0  # 到目前为止执行的时间步
        while t_so_far < total_timesteps:  # ppo的更新次数
            (
                batch_obs,
                batch_acts,
                batch_log_probs,
                batch_r,
                batch_r_to_go,
                batch_lens,
                batch_entropy,
            ) = self.rollout()

            if test < 1:
                print("batch_obs: ", batch_obs.shape)
                #                 print("batch_obs: ", batch_obs)
                print("batch_acts: ", batch_acts.shape)
                print("batch_log_probs: ", batch_log_probs.shape)
                print("batch_r[0]: ", len(batch_r[0]))
                #             print("batch_r: ", batch_r[0])
                print("batch_r_to_go: ", batch_r_to_go.shape)
                #             print("batch_lens: ", len(batch_lens))
                print("batch_lens: ", batch_lens)
                test += 1

            # calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # 更新 actor_sample

            self.actor_sample.load_state_dict(self.actor.state_dict())
            #             print("self.actor.state_dict(): ", self.actor.state_dict())

            # calculate V_{phi, k}
            V = self.evaluate(batch_obs)
            #             print("V: ", V.shape)

            #             self.V_record.append(V.detach().numpy())
            #             self.v_target_record.append(batch_r_to_go)

            # 检查critic网络
            # 存储timesteps的即时奖励
            #             self.timesteps_r_record.append(batch_r)
            #             self.critic_record.append(V)

            # calculate advantage
            A_k = batch_r_to_go - V.detach()
            #             print("A_k: ", A_k.shape)
            # trick: normalize advantage
            # A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                # epoch code
                # calculate V_phi and pi_theta(a_t | s_t)

                curr_mu, curr_sigma = self.actor(batch_obs)
                curr_dist = torch.distributions.Normal(curr_mu, curr_sigma)
                curr_log_probs = curr_dist.log_prob(batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # calculate surrogate losses

                batch_entropy = torch.tensor(batch_entropy, dtype=torch.float)
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                #                 actor_loss = (
                #                     -torch.min(surr1, surr2) - self.alpha * batch_entropy
                #                 ).mean()
                actor_loss = -torch.mean(torch.min(surr1, surr2))

                V = self.evaluate(batch_obs)
                v_target = batch_r_to_go

                #                 self.V_target.append(V)
                #                 self.v_target_record.append(v_target)

                critic_loss = nn.MSELoss()(V, v_target)

                # calculate gradients and perform backward propagation for actor/critic network
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()

                actor_loss.backward()
                critic_loss.backward()

                self.actor_optim.step()
                self.critic_optim.step()


#                 print("self.actor.state_dict(): ", self.actor.state_dict())


# In[92]:


model = PPO()
model.learn(40000)
env.close()
np.save("ppo.npy", model.reward_record)  # 保存为.npy格式
reward = np.load("ppo.npy")
reward = reward.tolist()
plt.plot(reward)
plt.show()


# In[ ]:





# In[ ]:





# In[25]:


np.save("ppo1.npy", model.V_record)  # 保存为.npy格式
V_record = np.load("ppo1.npy")
V_record = V_record.tolist()
plt.plot(V_record[150])
plt.show()


# In[26]:


np.save("ppo2.npy", model.v_target_record)  # 保存为.npy格式
v_target_record = np.load("ppo2.npy", allow_pickle=True)
v_target_record = v_target_record.tolist()
plt.plot(v_target_record[150])
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




