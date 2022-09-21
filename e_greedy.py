import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import time

EPISODE = 1000
Number_of_Bandits = 4
p_bandits = [0.1, 0.2, 0.4, 0.9]  # Probability of each bandit


class EpsilonGreedy:
    def __init__(self):
        self.epsilon = 0.1  # define "epsilon" value
        self.num_bandits = Number_of_Bandits  # number of bandits
        self.bandits = p_bandits  # each bandit probability, between 0~1
        self.best = np.argmax(self.bandits)  # index of the best bandit, which has the highest probability
        self.T = EPISODE  # number of steps for Epsilon Greedy
        self.hit = np.zeros(self.T)  # if action choose the best bandit, then hit +1
        self.reward = np.zeros(self.num_bandits)  # each bandit its "reward"
        self.num = np.zeros(self.num_bandits)  # each bandit its number of being selected

    def get_reward(self, i):  # i = index of bandit
        reward = self.bandits[i] + np.random.normal(0, 1)  # probability of bandit + random value between -1~1
        return reward

    def update(self, i):
        self.num[i] += 1
        self.reward[i] = (self.reward[i]*(self.num[i]-1)+self.get_reward(i))/self.num[i]

    def exploit(self):
        index = np.argmax(self.reward)
        return index

    def explore(self):
        a = np.argmax(self.reward)
        index = a
        while index == a:
            index = np.random.randint(0, self.num_bandits)
        return index

    def calculate(self):
        for i in range(self.T):
            if np.random.random() > self.epsilon:
                index = self.exploit()
            else:
                index = self.explore()

            if index == self.best:
                self.hit[i] = 1  # if selected bandit has the highest probability, hit +1
            self.update(index)

    def plot(self):
        x = np.array(range(self.T))
        y1 = np.zeros(self.T)
        y2 = np.ones(self.T)*(1-self.epsilon)
        t = 0
        for i in range(self.T):
            t += self.hit[i]
            y1[i] = t/(i+1)  # y = correct rate at step "i"
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(axis='x', color='0.80')
        plt.plot(x, y1, label='Epsilon')
        plt.plot(x, y2, label='1-Epsilon')
        plt.legend(title='Parameter where:')
        plt.show()


# main
np.random.seed(37)

# Epsilon Greedy
start_epsilon = time.time()
E = EpsilonGreedy()
E.calculate()
end_epsilon = time.time()
epsilon_time = end_epsilon - start_epsilon
E.plot()

print(f"Epsilon Greedy Time: {round(epsilon_time, 5)}(s)")

