import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import time

EPISODE = 1000
Number_of_Bandits = 4
p_bandits = [0.1, 0.2, 0.4, 0.9]  # Probability of each bandit


class UCB:
    def __init__(self):
        self.num_bandits = Number_of_Bandits  # number of bandits
        self.bandits = p_bandits  # each bandit probability, between 0~1
        self.best = np.argmax(self.bandits)  # index of the best bandit, which has the highest probability
        self.T = EPISODE  # number of steps for UCB
        self.hit = np.zeros(self.T)  # if action choose the best bandit, then hit +1
        self.reward = np.zeros(self.num_bandits)  # each bandit its "reward"
        self.num = np.ones(self.num_bandits) * 0.00001  # each bandit its number of being selected
        self.V = 0
        self.upper_bound = np.zeros(self.num_bandits)

    def get_reward(self, i):  # i = index of bandit
        return self.bandits[i] + np.random.normal(0, 1)  # probability of bandit + random value between -1~1

    def update(self, i):
        self.num[i] += 1
        self.reward[i] = (self.reward[i]*(self.num[i]-1)+self.get_reward(i))/self.num[i]
        self.V += self.get_reward(i)

    def calculate(self):
        for i in range(self.T):
            for j in range(self.num_bandits):
                self.upper_bound[j] = self.reward[j] + np.sqrt(2 * np.log(i + 1) / self.num[j])
            index = np.argmax(self.upper_bound)  # select the bandit which has the highest upper confidence bound
            if index == self.best:
                self.hit[i] = 1
            self.update(index)

    def plot(self):
        x = np.array(range(self.T))
        y = np.zeros(self.T)
        t = 0
        for i in range(self.T):
            t += self.hit[i]
            y[i] = t/(i+1)  # y = correct rate at step "i"
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(axis='x', color='0.80')
        plt.plot(x, y, label='UCB')
        plt.legend(title='Parameter where:')
        plt.show()


# main
np.random.seed(37)

# Upper Confidence Bound
start_ucb = time.time()
U = UCB()
U.calculate()
end_ucb = time.time()
ucb_time = end_ucb - start_ucb
U.plot()

print(f"UCB Time: {round(ucb_time, 5)}(s)")
