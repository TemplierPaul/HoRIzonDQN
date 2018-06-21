import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas as pd
from random import *

# Hyper Parameters
EPI_FILE = pd.read_csv("dataHorizon/out/up_2.csv")
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
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, N_NEURAL)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc2 = nn.Linear(N_NEURAL, N_NEURAL)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(N_NEURAL, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class SavedNet(object):

    def __init__(self):
        self.net = Net()
        net= torch.load('SavedNetwork/2018-06-20-15-34-46-Conv_CPU_target.pkl')

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


X_TABLE = 16
Y_TABLE = 16


def create_state():
    L = []
    L.append(randint (0, 600))          # time
    L.append(randint(0, 1))             # auto
    L.append((2*random()-1) * X_TABLE)  # x
    L.append((2*random()-1) * Y_TABLE)  # y
    L.append((2*random()-1) * 3.14)     # angle
    L.append(randint(0, 100))           # battery
    L.append(randint(50, 300))          # temperature
    L.append(randint(0, 100))           # robot tank
    L.append(randint(0, 100))           # ground tank
    for i in range (18):
        L.append(randint(0, 1))         # leaks, trees
    for k in range (10):                # keys
        a = randint(0, 6)
        for i in range(5):
                if (a == i):
                    L.append(1)
                else :
                    L.append(0)
    L.append(randint(0, 10))            # wrench
    L.append(randint(0, 10))            # minus
    L.append(randint(0, 10))            # plus
    L.append(randint(0, 10))            # push
    L.append(randint(0, 10))            # removeAlarms
    L.append(randint(0, 10))            # clickLeaks
    L.append(randint(0, 1))             # otherKeyUp
    L.append(randint(0, 1))             # otherKeyDown
    L.append(randint(0, 1))             # otherClick
    L.append(randint(0, 1))             # Keyboard
    return L


def check_accuracy(s, a):
    L = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if a == 0: # man to auto
        if s[1] != 0:
            L[a]=1
    if a == 1: # auto to man
        if s[1] != 1:
            L[a]=1
    if a == 2: # battery
        if s[5] > 50 :
            L[a] = 1
    if a == 3: # temperature
        if s[6] < 150 :
            L[a] = 1
    if a == 4: # time
        if s[0] > 120 :
            L[a] = 1
    if a == 5: # robot tank
        if s[7] > 50 :
            L[a] = 1
    if a == 6: # auto
        if s[1] != 1 :
            L[a] = 1
    if a == 7: # man
        if s[1] != 0 :
            L[a] = 1
    if a == 9: # main tank
        if s[8] > 50 :
            L[a] = 1
    return L


def main_test(n):
    dqn = SavedNet()
    log = open('log_test.txt', 'w')
    action_pb = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in range (n):
        L = create_state()
        s = Variable(torch.FloatTensor(L))
        Q = dqn(s)
        a = Q.index(max(Q))
        ver = check_accuracy(s, a)
        for j in range (len(action_pb)):
            action_pb[j] = action_pb[j] + ver[j]
        if sum(ver) != 0:
            log.write (str(L))
        print(i, '/', n, ":", str(sum(ver)))
    print (str(action_pb))
    log.write(str(action_pb))
    log.close()


N = 1000
main_test(N)