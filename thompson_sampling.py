import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import time

EPISODE = 1000
Number_of_Bandits = 4
p_bandits = [0.1, 0.2, 0.4, 0.9]  # Probability of each bandit


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


# main
np.random.seed(37)
ALGORITHM = ['Epsilon Greedy', 'UCB', 'Thompson Sampling']

# Thompson Sampling
start_thompson = time.time()
T = Thompson()
T.calculate()
end_thompson = time.time()
thompson_time = end_thompson - start_thompson
T.plot_rewards(T.average_reward)

print(f"Thompson Sampling Time: {round(thompson_time, 5)}(s)")

