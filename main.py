import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

EPISODE = 10000
Number_of_Bandits = 4
p_bandits = [0.5, 0.1, 0.8, 0.9]  # Probability of each bandit

class EpsilonGreedy:
    def __init__(self):
        self.epsilon = 0.1  # 设定epsilon值
        self.num_arm = Number_of_Bandits  # 设置arm的数量
        self.arms = p_bandits  # 设置每一个arm的均值，为0-1之间的随机数
        self.best = np.argmax(self.arms)  # 找到最优arm的index
        self.T = EPISODE  # 设置进行行动的次数
        self.hit = np.zeros(self.T)  # 用来记录每次行动是否找到最优arm
        self.reward = np.zeros(self.num_arm)  # 用来记录每次行动后各个arm的平均收益
        self.num = np.zeros(self.num_arm)  # 用来记录每次行动后各个arm被拉动的总次数

    def get_reward(self, i):  # i为arm的index
        return self.arms[i] + np.random.normal(0, 1)  # 生成的收益为arm的均值加上一个波动

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
            index = np.random.randint(0, self.num_arm)
        return index

    def calculate(self):
        for i in range(self.T):
            if np.random.random() > self.epsilon:
                index = self.exploit()
            else:
                index = self.explore()

            if index == self.best:
                self.hit[i] = 1  # 如果拿到的arm是最优的arm，则将其记为1
            self.update(index)

    def plot(self):  # 画图查看收敛性
        x = np.array(range(self.T))
        y1 = np.zeros(self.T)
        t = 0
        for i in range(self.T):
            t += self.hit[i]
            y1[i] = t/(i+1)
        y2 = np.ones(self.T)*(1-self.epsilon)
        plt.plot(x, y1)
        plt.plot(x, y2)
        plt.show()


class UCB:
    def __init__(self):
        self.num_arm = Number_of_Bandits  # 设置arm的数量
        self.arms = p_bandits  # 设置每一个arm的均值，为0-1之间的随机数
        self.best = np.argmax(self.arms)  # 找到最优arm的index
        self.T = EPISODE  # 设置进行行动的次数
        self.hit = np.zeros(self.T)  # 用来记录每次行动是否找到最优arm
        self.reward = np.zeros(self.num_arm)  # 用来记录每次行动后各个arm的平均收益
        self.num = np.ones(self.num_arm)*0.00001  # 用来记录每次行动后各个arm被拉动的总次数
        self.V = 0
        self.up_bound = np.zeros(self.num_arm)

    def get_reward(self, i):  # i为arm的index
        return self.arms[i] + np.random.normal(0, 1)  # 生成的收益为arm的均值加上一个波动

    def update(self, i):
        self.num[i] += 1
        self.reward[i] = (self.reward[i]*(self.num[i]-1)+self.get_reward(i))/self.num[i]
        self.V += self.get_reward(i)

    def calculate(self):
        for i in range(self.T):
            for j in range(self.num_arm):
                self.up_bound[j] = self.reward[j] + np.sqrt(2*np.log(i+1)/self.num[j])
            index = np.argmax(self.up_bound)
            if index == self.best:
                self.hit[i] = 1
            self.update(index)

    def plot(self):  # 画图查看收敛性
        x = np.array(range(self.T))
        y1 = np.zeros(self.T)
        t = 0
        for i in range(self.T):
            t += self.hit[i]
            y1[i] = t/(i+1)
        plt.plot(x, y1)
        plt.show()


class Thompson:
    def __init__(self):
        self.number_of_bandits = Number_of_Bandits
        self.p_bandits = p_bandits
        self.N = EPISODE  # number of steps for Thompson Sampling
        self.bandit_running_count = [1] * Number_of_Bandits  # Array for Number of bandits try times, e.g. [0. 0, 0]
        self.bandit_win_count = [0] * Number_of_Bandits  # Array for Number of bandits win times, e.g. [0. 0, 0]
        self.average_reward = []

    def bandit_run(self, index):
        if np.random.rand() >= self.p_bandits[index]:  # random  probability to win or lose per machine
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
                prob_theta_samples.append(p.rvs(1))  # rvs method provides random samples of distibution

            # Select best bandit based on theta sample a bandit
            select_bandit = np.argmax(prob_theta_samples)

            # Run bandit and get both win count and run count
            self.bandit_win_count[select_bandit] += self.bandit_run(select_bandit)
            self.bandit_running_count[select_bandit] += 1

            # if step == 3 or step == 11 or (step % 100 == 1 and step <= 1000) :
            #     plot_steps(bandit_distribution, step - 1, next(ax))

            # Elemtwise division of lists using zip() and create new list [AvgRewardARM1, AvgRewardARM2, AvgRewardARM3, ...]
            # We do bandit_win_count[1]+bandit_win_count[2]+...) / (bandit_runing_count[0] + ...
            average_reward_list = ([n / m for n, m in zip(self.bandit_win_count, self.bandit_running_count)])

            # Get average of all bandits into only one reward value
            averaged_total_reward = 0
            for avged_arm_reward in (average_reward_list):
                averaged_total_reward += avged_arm_reward
            self.average_reward.append(averaged_total_reward)

    def plot_rewards(self, rewards):
        # plt.figure(2)
        plt.title('Average Reward Comparison')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(rewards, color='green', label='Thompson')
        plt.grid(axis='x', color='0.80')
        plt.legend(title='Parameter where:')
        plt.show()


def plot_compare(hit_epsilon, hit_ucb, hit_thompson):
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
    plt.show()


np.random.seed(23)
E = EpsilonGreedy()
E.calculate()
E.plot()

U = UCB()
U.calculate()
U.plot()

T = Thompson()
T.calculate()
T.plot_rewards(T.average_reward)

plot_compare(hit_epsilon=E.hit, hit_ucb=U.hit, hit_thompson=T.average_reward)

