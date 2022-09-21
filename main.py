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


class Thompson:
    def __init__(self):
        self.number_of_bandits = Number_of_Bandits
        self.p_bandits = p_bandits
        self.N = EPISODE  # number of steps for Thompson Sampling
        self.bandit_running_count = [1] * Number_of_Bandits  # Array for Number of bandits try times, e.g. [0, 0, 0, 0]
        self.bandit_win_count = [0] * Number_of_Bandits  # Array for Number of bandits win times, e.g. [0, 0, 0, 0]
        self.average_reward = []

    def bandit_run(self, index):
        if np.random.rand() >= self.p_bandits[index]:  # random probability to win or lose per machine
            return 0  # Lose
        else:
            return 1  # Win

    def calculate(self):
        for step in range(1, self.N):
            # Beta distribution and alfa beta calculation
            bandit_distribution = []
            for run_count, win_count in zip(self.bandit_running_count, self.bandit_win_count):  # create a tuple() of count and win
                bandit_distribution.append(stats.beta(a=win_count + 1,
                                                      b=run_count - win_count + 1))  # calculate the main equation (beta distribution)

            prob_theta_samples = []
            # Theta probability sampling for each bandit
            for p in bandit_distribution:
                prob_theta_samples.append(p.rvs(1))  # rvs method provides random samples of distribution

            # Select best bandit based on theta sample a bandit
            select_bandit = np.argmax(prob_theta_samples)

            # Run bandit and get both win count and run count
            self.bandit_win_count[select_bandit] += self.bandit_run(select_bandit)
            self.bandit_running_count[select_bandit] += 1

            # Elemtwise division of lists using zip() and create new list [AvgRewardARM1, AvgRewardARM2, AvgRewardARM3, ...]
            # We do bandit_win_count[1]+bandit_win_count[2]+...) / (bandit_running_count[0] + ...
            average_reward_list = ([n / m for n, m in zip(self.bandit_win_count, self.bandit_running_count)])

            # Get average of all bandits into only one reward value
            averaged_total_reward = 0
            for averaged_arm_reward in average_reward_list:
                averaged_total_reward += averaged_arm_reward
            self.average_reward.append(averaged_total_reward)

    def plot_rewards(self, rewards):
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(rewards, color='green', label='Thompson')
        plt.grid(axis='x', color='0.80')
        plt.legend(title='Parameter where:')
        plt.show()


def plot_reward_compare(hit_epsilon, hit_ucb, hit_thompson):
    x = np.array(range(EPISODE))
    y_epsilon = np.zeros(EPISODE)
    y_ucb = np.zeros(EPISODE)
    t = 0
    for i in range(EPISODE):
        t += hit_epsilon[i]
        y_epsilon[i] = t / (i + 1)
    t = 0
    for i in range(EPISODE):
        t += hit_ucb[i]
        y_ucb[i] = t / (i + 1)

    plt.title('Average Reward Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(x, y_epsilon, color='red', label='Epsilon-Greedy')
    plt.plot(x, y_ucb, color='blue', label='UCB')
    plt.plot(hit_thompson, color='green', label='Thompson')
    plt.legend(title='Parameter where:')
    plt.show()


def plot_time_compare(algorithm_name_1, time_1, algorithm_name_2, time_2, algorithm_name_3, time_3):
    plt.title('Time Comparison')
    plt.xlabel('Algorithm')
    plt.ylabel('Time')
    plt.bar(algorithm_name_1, time_1, width=0.8, bottom=None, align='center', color='red')
    plt.bar(algorithm_name_2, time_2, width=0.8, bottom=None, align='center', color='blue')
    plt.bar(algorithm_name_3, time_3, width=0.8, bottom=None, align='center', color='green')
    plt.show()


# main
np.random.seed(37)
ALGORITHM = ['Epsilon Greedy', 'UCB', 'Thompson Sampling']

# Epsilon Greedy
start_epsilon = time.time()
E = EpsilonGreedy()
E.calculate()
end_epsilon = time.time()
epsilon_time = end_epsilon - start_epsilon
# E.plot()

# Upper Confidence Bound
start_ucb = time.time()
U = UCB()
U.calculate()
end_ucb = time.time()
ucb_time = end_ucb - start_ucb
# U.plot()

# Thompson Sampling
start_thompson = time.time()
T = Thompson()
T.calculate()
end_thompson = time.time()
thompson_time = end_thompson - start_thompson
# T.plot_rewards(T.average_reward)

print(f"Epsilon Greedy Time: {round(epsilon_time, 5)}(s)")
print(f"UCB Time: {round(ucb_time, 5)}(s)")
print(f"Thompson Sampling Time: {round(thompson_time, 5)}(s)")

plot_reward_compare(hit_epsilon=E.hit, hit_ucb=U.hit, hit_thompson=T.average_reward)
plot_time_compare(ALGORITHM[0], epsilon_time, ALGORITHM[1], ucb_time, ALGORITHM[2], thompson_time)

