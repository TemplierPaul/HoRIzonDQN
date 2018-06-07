import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas as pd


# Hyper Parameters
EPI_FILE = pd.read_csv("dataHorizon/out/up_0.csv")
N_ACTIONS = 10
N_STATES = EPI_FILE.columns.size - N_ACTIONS - 1
print("N_ACTIONS:", N_ACTIONS)
BATCH_SIZE = 32   # Number of samples selected per study
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 10   # target update frequency
MEMORY_CAPACITY = 500
N_EPISODE=2000   #Number of files read (number of experiments)
N_EXP_TOL=400    #If the game is running too long, go to the next experiment(temporarily not considered)


class Net(nn.Modul):
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
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 1 + 1))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        action=np.array(EPI_FILE.ix[x, N_STATES :N_STATES+N_ACTIONS])
        action_index=8   # 8 means alarm 6, means Non
        for i in range(N_ACTIONS):
            if(action[i]>0):
                action_index=i
                break
        return action_index

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
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
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+1+1]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))
        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a )# shape (batch, 1)//value of the chosen action
        #print("q_eval",q_eval)
        q_eval4action = self.eval_net(b_s_).detach()     # detach from graph, don't backpropagate/value of all the actions
        #print("q_eval4action=",q_eval4action)
        # the action that brings the highest value is evaluated by q_eval
        (actions_max,actions_max_index)=torch.max(q_eval4action,1)
        #print("action_max_index=", actions_max_index)
        actions_max_index=torch.unsqueeze(actions_max_index, 1)
        #print("action_max_index=",actions_max_index)
        q_next=self.target_net(b_s_).gather(1,actions_max_index).detach()  # detach from graph, don't backpropagate
        #print("q_next=", q_next)
        #print("b_r=", b_r)
        q_target = b_r + GAMMA * q_next  # shape (batch, 1)
        #print("q_target=",q_target)
        loss = self.loss_func(q_eval, q_target)
        #print("loss=",loss)

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
    s_i = 0
    s=np.array(EPI_FILE.ix[s_i, 0:N_STATES])
    ep_r = 0
    while (s_i<N_EXP-1):



        # take action
        a_index = dqn.choose_action(s_i)

        r = np.array(EPI_FILE.ix[s_i, N_STATES+N_ACTIONS])
        s_i=s_i+1
        # if s_i>N_EXP_TOL:
        #     break
        s_next = np.array(EPI_FILE.ix[s_i, 0:N_STATES])


        dqn.store_transition(s, a_index, r, s_next)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            print('Ep: ', i_episode,
                '| Ep_r: ', round(ep_r, 2))
            #print("weight=",dqn.target_net.fc1.weight)

        s = s_next

torch.save(dqn.eval_net, 'DoubleDQN_eval_net.pkl')
torch.save(dqn.target_net, 'DoubbleDQN_target_net.pkl')
