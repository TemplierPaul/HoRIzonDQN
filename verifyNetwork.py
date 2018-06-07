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
MEMORY_CAPACITY = 1000
N_EPISODE=2000   #Number of files read (number of experiments)
N_EXP_TOL=400    #If the game is running too long, go to the next experiment(temporarily not considered)

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

class SavedNet(object):

    def __init__(self):
        self.net = Net()
        net= torch.load('DQN_target_net.pkl')

    def orig_action(self, x):
        action=np.array(EPI_FILE.ix[x, N_STATES :N_STATES+N_ACTIONS])
        action_index=8  # 8 means alarm 6, means None
        for i in range(N_ACTIONS):
            #print("i=",i)
            if(action[i]>0):
                action_index=i
                break
        return action_index

    def choose_action(self,s):
        s = Variable(torch.FloatTensor(s))
        q_next = self.net(s).detach()  # detach from graph, don't backpropagate/value of all the actions
        print(q_next)
        #q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)



NetToVerify=SavedNet()

str_filename="verify.csv"
EPI_FILE = pd.read_csv(str_filename)

N_EXP = EPI_FILE.iloc[:, 0].size
s_i = 0
s=np.array(EPI_FILE.ix[s_i, 0:N_STATES])
ep_r = 0
while (s_i<N_EXP-1):

    # take action
    a_orig = NetToVerify.orig_action(s_i)
    print("a_orig=",a_orig)
    NetToVerify.choose_action(s)
    r = np.array(EPI_FILE.ix[s_i, N_STATES+N_ACTIONS])
    s_i=s_i+1
    # if s_i>N_EXP_TOL:
    #     break
    s_next = np.array(EPI_FILE.ix[s_i, 0:N_STATES])


    ep_r += r


    s = s_next

