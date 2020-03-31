import time
import numpy as np
import pandas as pd


ACTIONS = ["left", "right"]  # 可选动作
EPSILON = 0.1  # 贪婪系数
ALPHA = 0.1  # 学习系数
GAMMA = 0.9  # 衰减系数
FRESH_TIME = 0.05  # 每次跳动时间(s)


class RLGame:
    def __init__(self, length: int, actions=None,
                 epsilon=EPSILON, alpha=ALPHA, gamma=GAMMA, fresh_time=FRESH_TIME):
        """
        :param length: 游戏长度
        :param actions: 动作列表，默认使用["left", "right"]
        :param epsilon: 贪婪系数
        :param alpha: 学习系数
        :param gamma: 衰减系数
        :param fresh_time: 每次跳动时间(s)
        """
        if actions is None:
            actions = ["left", "right"]
        self.QTable = pd.DataFrame(
            np.zeros((length, len(actions))),
            columns=actions
        )
        self.rewords = [0] * (length - 1) + [1]
        np.random.seed(int(time.time()))
        self.state = 0
        self.stepCount = 0
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.freshTime = fresh_time
        self.length = length
        self.actions = actions

    def is_terminated(self):
        """
        :return: 当前回合是否结束
        """
        return self.state == self.length - 1

    def is_terminated_state(self, state):
        """
        :param state: 状态号
        :return: 该状态号是否是终态
        """
        return state == self.length - 1

    def learn(self, to_print=False):
        """
        一个回合的学习过程
        :param to_print: 是否打印过程信息
        :return: 该回合学习后的Q table
        """
        while not self.is_terminated():
            if to_print:
                self.print()
            self.stepCount += 1
            action = self.choose_action()
            q_predict = self.QTable.loc[self.state, action]
            next_state, reward = self.observe(action)
            q_real = self.QTable.iloc[next_state, :].max()
            self.QTable.loc[self.state, action] +=\
                self.alpha * (reward + self.gamma * q_real - q_predict)
            self.state = next_state
        if to_print:
            print("\rFinish")
            print("Total Steps = %s" % self.stepCount)
        return self.QTable

    def choose_action(self):
        """
        :return: 当前状态下通过ε-贪婪策略选择动作
        """
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            index = self.QTable.iloc[self.state, :].index
            candidate_actions = self.QTable.take(np.random.permutation(len(index)), axis=1).iloc[self.state, :]
            action = candidate_actions.idxmax()
        return action

    def observe(self, action):
        """
        :param action: 采取的动作
        :return: 采取该动作后的下一个状态和回报
        """
        if action == "left":
            next_state = max(self.state - 1, 0)
        else:
            next_state = min(self.state + 1, self.length - 1)
        reward = self.rewords[next_state]
        return next_state, reward

    def print(self):
        """
        打印当前游戏的信息
        """
        # 得到的是['-', '-', '-', 'T']这样的数组
        env_list = ["-"] * (self.length - 1) + ["T"]
        env_list[self.state] = 'o'
        msg = ''.join(env_list)
        print("\r{}".format(msg), end='')
        time.sleep(self.freshTime)

    def reset(self):
        self.state = 0
        self.stepCount = 0
