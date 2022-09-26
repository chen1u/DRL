import numpy as np
import matplotlib.pyplot as plt

reward1 = np.load('dqn.npy')
reward1 = reward1.tolist()
reward2 = np.load('ddqn.npy')
reward2 = reward2.tolist()
reward3 = np.load('duelingdqn.npy')
reward3 = reward3.tolist()
reward4 = np.load('d3qn.npy')
reward4 = reward4.tolist()
reward5 = np.load('ddqn_per.npy')
reward5 = reward5.tolist()

# print(len(reward1))
# print(len(reward2))
# print(len(reward3))
# print(len(reward4))

plt.plot(reward1, label='dqn')
plt.plot(reward2, label='ddqn')
plt.plot(reward3, label='duelingdqn')
plt.plot(reward4, label='d3qn')
plt.plot(reward5, label='ddqn_per')
plt.legend()
plt.show()