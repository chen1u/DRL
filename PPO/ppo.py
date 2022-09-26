import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
from torch.optim import Adam

# 参考：
# https://blog.csdn.net/Azahaxia/article/details/117329002
# https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-3-4-82081ea58146

env = gym.make('Pendulum-v1')
# env = env.unwrapped
N_ACTIONS = env.action_space.shape[0]
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, out_dim)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)
        return output


class PPO:
    def __init__(self):
        self._init_hyperparameters()

        self.actor = FeedForwardNN(N_STATES, N_ACTIONS)
        self.critic = FeedForwardNN(N_STATES, 1)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # create our variable for the matrix
        # chose 0.5 for stdev arbitrarily
        self.cov_var = torch.full(size=(N_ACTIONS,), fill_value=0.5)

        # create the covariance matrix
        self.cov_var = torch.diag(self.cov_var)

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 2048
        self.max_timesteps_per_episode = 1600
        self.gamma = 0.99
        self.n_updates_per_iteration = 10
        self.clip = 0.2
        self.lr = 3e-4
        self.i_episode = 0

    def get_action(self, obs):
        # for a mean action
        # same thing as calling self.action.forward(obs)
        mean = self.actor(obs)

        # create Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_var)

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
        return action.detach().numpy(), log_prob.detach()

    def compute_rtgs(self, batch_r):
        # the rewards-to-go (rtg) per episode per batch to return
        # the shape will be (num timesteps per episode)
        batch_r_to_go = []
        for ep_r in reversed(batch_r):
            discounted_r = 0
            for r in reversed(ep_r):
                discounted_r = r +discounted_r * self.gamma
                batch_r_to_go.insert(0, discounted_r)

        batch_r_to_go = torch.tensor(batch_r_to_go, dtype=torch.float)
        return batch_r_to_go

    def rollout(self):
        # batch data
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_r = []
        batch_r_to_go = []
        batch_lens = []  # episodic lengths in batch

        # Number of timesteps run so far this batch
        t = 0

        while t < self.timesteps_per_batch:
            # rewards of this episode
            ep_r = []
            obs = env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                env.render()
                t += 1
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, r, done, _ = env.step(action)
                ep_r.append(r)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    self.i_episode += 1
                    ep_r_sum = sum(ep_r)
                    print('Ep: ', self.i_episode,
                          '| Ep_r: ', round(ep_r_sum))
                    reward_record.append(ep_r_sum)
                    break

            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1)  # plus 1 because timestep starts at 0
            batch_r.append(ep_r)

            batch_obs = np.array(batch_obs)
            batch_acts = np.array(batch_acts)
            batch_log_probs = np.array(batch_log_probs)
            batch_obs = torch.tensor(batch_obs, dtype=torch.float)
            batch_acts = torch.tensor(batch_acts, dtype=torch.float)
            batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

            batch_r_to_go = self.compute_rtgs(batch_r)

            return batch_obs, batch_acts, batch_log_probs, batch_r_to_go, batch_lens

    def evaluate(self, batch_obs, batch_acts):
        # query critic network for a value V for each obs in batch_obs
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most
        # recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_var)
        # calculate Ratio of action probabilities
        # with parameters Θ over action probabilities with parameters Θₖ
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs

    def learn(self, total_timesteps):
        t_so_far = 0 # 到目前为止执行的时间步
        while t_so_far < total_timesteps:  # ppo的更新次数
            batch_obs, batch_acts, batch_log_probs, batch_r_to_go, batch_lens = self.rollout()

            # calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)


            # calculate V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_acts)
            # calculate advantage
            A_k = batch_r_to_go - V.detach()
            # trick: normalize advantage
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                # epoch code
                # calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_r_to_go)

                # calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()


reward_record = []
model = PPO()
model.learn(40000)
env.close()
np.save('ppo.npy', reward_record)  # 保存为.npy格式
reward = np.load('ppo.npy')
reward = reward.tolist()
plt.plot(reward)
plt.show()
                





