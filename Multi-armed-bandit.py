# -*- coding: utf-8 -*-
"""
E3A - 456

The content of this program has been largely taken from

https://towardsdatascience.com 

Last update 27/10/2021
"""

# import modules 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

from eps_bandit import eps_bandit

# Main part of the program
# Number of arms
k = 10
# Number of iterations in each episode
iters = 10000
eps_1_selection = np.zeros(k)
eps_2_selection = np.zeros(k)
eps_3_selection = np.zeros(k)

# Array to store the rewards
eps_1_rewards = np.zeros(iters)
eps_2_rewards = np.zeros(iters)
eps_3_rewards = np.zeros(iters)

# Number of episodes for the simulation 
episodes = 100

# Run experiments
for i in range(episodes):
    # Initialize bandits
    eps_1 = eps_bandit(k, 0.1, iters, mu='sequence')
    eps_2 = eps_bandit(k, 0, iters, mu='sequence')
    eps_3 = eps_bandit(k, 0.01, iters, mu='sequence')
    
    # Run experiments
    eps_1.run()
    eps_2.run()
    eps_3.run()
    
    # Update long-term averages among episodes
    eps_1_rewards = eps_1_rewards + (
        eps_1.reward - eps_1_rewards) / (i + 1)
    eps_1_selection = eps_1_selection + (
        eps_1.k_n - eps_1_selection) / (i + 1)
    eps_2_rewards = eps_2_rewards + (
            eps_2.reward - eps_2_rewards) / (i + 1)
    eps_2_selection = eps_2_selection + (
        eps_2.k_n - eps_2_selection) / (i + 1)
    eps_3_rewards = eps_3_rewards + (
            eps_3.reward - eps_3_rewards) / (i + 1)
    eps_3_selection = eps_3_selection + (
        eps_3.k_n - eps_3_selection) / (i + 1)

plt.figure(figsize=(14,8))
plt.plot(eps_1_rewards, label="$\epsilon=0.1$")
plt.plot(eps_2_rewards, label="$\epsilon=0$")
plt.plot(eps_3_rewards, label="$\epsilon=0.01$")
plt.legend(bbox_to_anchor=(0.6, 0.5))
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.title("Average $\epsilon$-greedy and greedy rewards after " + str(episodes) 
    + " Episodes")
plt.show()
bins = np.linspace(0, k-1, k)
plt.figure(figsize=(12,8))
plt.bar(bins, eps_2_selection,
width = 0.33, color='b', label="$\epsilon=0$")
plt.bar(bins+0.33, eps_3_selection,
width=0.33, color='g', label="$\epsilon=0.01$")
plt.bar(bins+0.66, eps_1_selection,
width=0.33, color='r', label="$\epsilon=0.1$")
plt.legend(bbox_to_anchor=(1.2, 0.5))
plt.xlim([0,k])
plt.title("Actions Selected by Each Algorithm")
plt.xlabel("Action")
plt.ylabel("Number of Actions Taken")
plt.show()
opt_per = np.array([eps_2_selection, eps_3_selection,
eps_1_selection]) / iters * 100
df = pd.DataFrame(opt_per, index=['$\epsilon=0$',
'$\epsilon=0.01$', '$\epsilon=0.1$'],
columns=["a = " + str(x) for x in range(0, k)])
print("Percentage of actions selected:")
df
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate εₙ based on the decay formula
def calculate_epsilon(beta, n):
    return 1 / (1 + beta * n)

# Number of iterations within an episode
iters = 1000
# Range of β values to visualize
beta_values = [0.01, 0.1, 0.5]

# Plotting
plt.figure(figsize=(10, 6))

for beta in beta_values:
    # Calculate εₙ for each iteration using the specified β
    epsilon_values = [calculate_epsilon(beta, n) for n in range(iters)]

    # Plot the evolution of εₙ for the current β
    plt.plot(range(iters), epsilon_values, label=f'β = {beta}')

plt.xlabel('Iteration (n)')
plt.ylabel('εₙ')
plt.title('Evolution of εₙ for Different β Values')
plt.legend()
plt.show()
class eps_decay_bandit:
    def __init__(self, k, epsilon_start, beta, iters, mu='sequence'):
        self.k = k
        self.epsilon_start = epsilon_start
        self.beta = beta
        self.iters = iters
        self.mu = mu
        self.k_n = np.zeros(k)
        self.reward = np.zeros(iters)
        self.arm_rewards = np.zeros(k)
        self.arm_counts = np.zeros(k)
        self.arm_means = np.zeros(k)

    def get_action(self):
        if np.random.random() < self.epsilon_start:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.arm_means)

    def update(self, action, reward):
        self.arm_rewards[action] += reward
        self.arm_counts[action] += 1
        self.arm_means[action] = self.arm_rewards[action] / self.arm_counts[action]

    def run(self):
        for t in range(self.iters):
            action = self.get_action()
            reward = self.mu(action)
            self.update(action, reward)
            self.reward[t] = reward
            self.k_n[action] += 1
            self.epsilon_start = 1 / (1 + self.beta * (t + 1))


class EpsilonGreedyBandit:
    def __init__(self, k, epsilon, iters):
        self.k = k
        self.epsilon = epsilon
        self.iters = iters
        self.k_n = np.zeros(k)
        self.k_reward = np.zeros(k)
        self.total_reward = 0
        self.average_reward = np.zeros(iters)

    def get_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.k_reward)

    def update(self, action, reward):
        self.k_n[action] += 1
        self.total_reward += reward
        self.k_reward[action] += (reward - self.k_reward[action]) / self.k_n[action]

    def run(self):
        for t in range(self.iters):
            action = self.get_action()
            reward = self.mu(action)
            self.update(action, reward)
            self.average_reward[t] = self.total_reward / (t + 1)


k = 10
iters = 1000
episodes = 100

epsilons = [0.1, 0.01, 0]  # Values of ε for comparison
beta_values = [0.1, 0.01, 0.001]  # Values of β for decay

rewards_no_decay = np.zeros(iters)
rewards_decay = np.zeros((len(beta_values), iters))

for i in range(episodes):
    # Without decay
    eps_no_decay = EpsilonGreedyBandit(k, epsilons[0], iters)
    eps_no_decay.run()
    rewards_no_decay += (eps_no_decay.average_reward - rewards_no_decay) / (i + 1)

    # With decay
    for j, beta in enumerate(beta_values):
        eps_decay = EpsilonGreedyBandit(k, epsilons[0], iters)
        eps_decay.epsilon = epsilons[0] / (1 + beta * np.arange(iters))
        eps_decay.run()
        rewards_decay[j] += (eps_decay.average_reward - rewards_decay[j]) / (i + 1)

# Plot average rewards
plt.plot(rewards_no_decay, label='No Decay')
for j, beta in enumerate(beta_values):
    plt.plot(rewards_decay[j], label=f'Decay (β = {beta})')
plt.xlabel('Iterations')
plt.ylabel('Average Reward')
plt.title('Comparison of Epsilon-Greedy Policies')
plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt


class eps_bandit:
    def __init__(self, k, epsilon, iters, mu='sequence'):
        self.k = k
        self.epsilon = epsilon
        self.iters = iters
        self.mu = mu
        self.k_n = np.zeros(k)
        self.reward = np.zeros(iters)
        self.arm_rewards = np.zeros(k)
        self.arm_counts = np.zeros(k)
        self.arm_means = np.zeros(k)

    def get_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.arm_means)

    def update(self, action, reward):
        self.arm_rewards[action] += reward
        self.arm_counts[action] += 1
        self.arm_means[action] = self.arm_rewards[action] / self.arm_counts[action]

    def run(self):
        for t in range(self.iters):
            action = self.get_action()
            reward = self.mu(action)
            self.update(action, reward)
            self.reward[t] = reward


class eps_decay_bandit:
    def __init__(self, k, epsilon_start, beta, iters, mu='sequence'):
        self.k = k
        self.epsilon_start = epsilon_start
        self.beta = beta
        self.iters = iters
        self.mu = mu
        self.k_n = np.zeros(k)
        self.reward = np.zeros(iters)
        self.arm_rewards = np.zeros(k)
        self.arm_counts = np.zeros(k)
        self.arm_means = np.zeros(k)

    def get_action(self):
        if np.random.random() < self.epsilon_start:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.arm_means)

    def update(self, action, reward):
        self.arm_rewards[action] += reward
        self.arm_counts[action] += 1
        self.arm_means[action] = self.arm_rewards[action] / self.arm_counts[action]

    def run(self):
        for t in range(self.iters):
            action = self.get_action()
            reward = self.mu(action)
            self.update(action, reward)
            self.reward[t] = reward
            self.k_n[action] += 1
            self.epsilon_start = 1 / (1 + self.beta * (t + 1))
k = 10
iters = 10000
episodes = 100
epsilons = [0.1, 0.01, 0]  # Values of ε for comparison
beta_values = [0.1, 0.01, 0.001]  # Values of β for decay
rewards_no_decay = np.zeros(iters)
rewards_decay = np.zeros(iters)
rewards_optimistic = np.zeros(iters)
for i in range(episodes):
    # Without decay
    eps_no_decay = eps_bandit(k, epsilons[0], iters, mu='sequence')
    eps_no_decay.run()
    rewards_no_decay += (eps_no_decay.reward - rewards_no_decay) / (i + 1)

    # With decay
    for beta in beta_values:
        eps_decay = eps_decay_bandit(k, epsilons[0], beta, iters, mu='sequence')
        eps_decay.run()
        rewards_decay += (eps_decay.reward - rewards_decay) / (i + 1)

    # Optimistic initialization
    oiv_bandit = eps_bandit(k, 0, iters, mu='sequence')
    oiv_bandit.arm_means = np.repeat(5., k)
    oiv_bandit.arm_counts = np.ones(k)
    oiv_bandit.run()
    rewards_optimistic += (oiv_bandit.reward - rewards_optimistic) / (i + 1)
plt.figure(figsize=(12, 8))
plt.plot(rewards_no_decay, label='No Decay')
for i, beta in enumerate(beta_values):
    plt.plot(rewards_decay, label=f'Decay (β = {beta})')
plt.plot(rewards_optimistic, label='Optimistic Initialization')
plt.xlabel('Iterations')
plt.ylabel('Average Reward')
plt.title('Comparison of ε-Greedy Policies')
plt.legend()
plt.show()
def get_action(self):
    if np.random.random() < self.epsilon:
        return np.random.randint(self.k)
    else:
        return np.argmax(self.k_reward)

def update(self, action, reward):
    self.k_n[action] += 1
    self.total_reward += reward
    self.k_reward[action] += (reward - self.k_reward[action]) / self.k_n[action]

def run(self):
    for t in range(self.iters):
        action = self.get_action()
        reward = self.mu(action)
        self.update(action, reward)
        self.average_reward[t] = self.total_reward / (t + 1)





