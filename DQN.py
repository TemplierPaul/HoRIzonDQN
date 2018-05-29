import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas as pd


# Hyper Parameters
EPI_FILE = pd.read_csv('test.csv')
N_ACTIONS = 16
N_STATES = EPI_FILE.columns.size - 3
print("N_ACTIONS:", N_ACTIONS)
BATCH_SIZE = 10
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 10   # target update frequency
MEMORY_CAPACITY = 100
N_EPISODE=100
N_EXP_TOL=200 #


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        action=np.array(EPI_FILE.ix[x, 1])
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))
        #print("b_memory=", b_memory)
        #print("b_a=",b_a)
        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)//value of the chosen action
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate/value of all the actions
        print("Q_next=",q_next)
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()

print('\nCollecting experience...')
for i_episode in range(N_EPISODE):
    str_filename="dataHorizon/out/up_"+str(i_episode)+".csv"
    try:
        EPI_FILE = pd.read_csv(str_filename)
    except FileNotFoundError:
        continue
    N_EXP = EPI_FILE.iloc[:, 0].size
    s_i = 1
    s=np.array(EPI_FILE.ix[s_i, 2:N_STATES+2])
    # print("s_i=",s_i)
    # print("s=",EXP_FILE.ix[s_i, 2:N_STATES+2])
    ep_r = 0
    while (s_i<N_EXP-1):

        a = dqn.choose_action(s_i)

        # take action
        # print("s_i=", s_i)
        # print("a=", a)
        # print("s=", EXP_FILE.ix[s_i, 2:N_STATES + 2])

        r = np.array(EPI_FILE.ix[s_i, N_STATES+2])
        s_i=s_i+1
        if s_i>N_EXP_TOL:
            break
        s_next = np.array(EPI_FILE.ix[s_i, 2:N_STATES+2])
        # modify the reward


        dqn.store_transition(s, a, r, s_next)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            print('Ep: ', i_episode,
                '| Ep_r: ', round(ep_r, 2))
            print("weight=",dqn.target_net.fc1.weight)

        s = s_next
