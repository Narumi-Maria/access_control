import numpy as np
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

def get_action(customer, server, value_function,EPSILON=0.1):
    if server == 0: #没有空闲服务器
        return 0
    #EPSILON=0.1 #ε贪心
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice([0,1]) #随机走一步
    if value_function.value(customer, server,1) > value_function.value(customer, server,0):
        return 1
    return 0

def step(customer, server, action):
    p=0.06
    if action == 0:
        return server + np.random.binomial((10-server), p)
    if server != 10:
        return server-1 + np.random.binomial((10-server-1), p)
    else:
        return 9

def semi_gradient_n_step_sarsa(value_function, n=1):
    #初始化状态S、动作A和average_reward
    customer_list=[1,2,4,8]
    current_customer = np.random.choice(customer_list)
    current_server = 10
    current_action = get_action(current_customer, current_server, value_function)
    average_reward = 0
    beta=0.01

    for i in tqdm(range(int(2e6))):
        #采取动作A,观察S‘和R
        new_customer = np.random.choice(customer_list)
        new_server = step(current_customer, current_server, current_action)
        current_reward = current_customer * current_action
        '''
                if new_server == 0:
            new_action = 0
        else:
            # 通过模型选取A’并训练模型
            new_action = get_action(new_customer, new_server, value_function)
            delta = current_reward - average_reward + value_function.value(new_customer, new_server,
                                                                           new_action) - value_function.value(
                current_customer, current_server, current_action)
            average_reward = average_reward + delta * beta
            value_function.learn(current_customer, current_server, current_action, delta)
        '''
        new_action = get_action(new_customer, new_server, value_function)
        delta = current_reward - average_reward + value_function.value(new_customer, new_server,
                                                                       new_action) - value_function.value(
            current_customer, current_server, current_action)
        average_reward = average_reward + delta * beta
        value_function.learn(current_customer, current_server, current_action, delta)

        current_customer = new_customer
        current_server = new_server
        current_action = new_action

    return average_reward

class ValueFunction:
    def __init__(self, step_size):
        self.step_size=step_size #alpha
        self.weights = np.zeros((4,11,2)) #表格形价值函数？我猜
        self.dic = {1: 0, 2: 1, 4: 2, 8: 3}

    def value(self, customer, server, action):
        return self.weights[self.dic[customer]][server][action]

    def learn(self,customer, server, action,delta):
        self.weights[self.dic[customer]][server][action] += self.step_size * delta

def figure_10_1():
    alpha = 0.01
    value_function = ValueFunction(alpha)
    semi_gradient_n_step_sarsa(value_function)

    policy = np.zeros((4, 11))
    for priority in range(4):
        for free_servers in range(11):
            policy[priority, free_servers] = get_action(pow(2, priority), free_servers, value_function,0)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 1, 1)
    fig = sns.heatmap(policy, cmap="YlGnBu", ax=ax, xticklabels=range(11), yticklabels=[1,2,4,8])
    fig.set_title('Policy (0 Reject, 1 Accept)')
    fig.set_xlabel('Number of free servers')
    fig.set_ylabel('Priority')
    plt.show()

if __name__ == '__main__':
    figure_10_1()