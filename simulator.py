import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math

# Hyper Parameters
EPI_FILE = pd.read_csv("dataHorizon/out/up_0.csv")
N_ACTIONS = 10
N_STATES = EPI_FILE.columns.size - N_ACTIONS - 1
N_STATES_REDUITS = 4
print("N_STATES:", N_STATES)
print("N_STATES_REDUIT:", N_STATES_REDUITS)
print("N_ACTIONS:", N_ACTIONS)
BATCH_SIZE = 32  # Number of samples selected per study
LR = 0.01  # learning rate
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 10  # target update frequency
MEMORY_CAPACITY = 1000
N_EPISODE = 2000  # Number of files read (number of experiments)
N_EXP_TOL = 400  # If the game is running too long, go to the next experiment(temporarily not considered)
N_NEURAL = 32
N_EPISODE = 1100
df = pd.read_csv('dataHorizon/out/outReduit/dataSet/dataSet.csv')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 32)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc2 = nn.Linear(32, 16)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(16, N_ACTIONS)
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
        self.net = torch.load('SavedNetwork/2018-06-20-09-46-32DoubbleDQN_target_net.pkl')
        print(self.net)
        # self.net = Net()

    def orig_action(self, x):
        action = np.array(EPI_FILE.ix[x, N_STATES:N_STATES + N_ACTIONS])
        action_index = 8  # 8 means alarm 6, means None
        for i in range(N_ACTIONS):
            # print("i=",i)
            if (action[i] > 0):
                action_index = i
                break
        return action_index

    def choose_action(self, s):
        sVar = Variable(torch.FloatTensor(s))
        #print("state=", sVar)
        q_next = self.net(sVar).detach()  # detach from graph, don't backpropagate/value of all the actions
        q_next = q_next.numpy()
        #print("Q-values=", q_next)
        #print("Chosen Action=", np.argmax(q_next))
        return np.argmax(q_next)

    def state_transition(self, s):
        #print("state to reduit", s)
        # print(s[1],s[2],s[3],s[4])
        sArr = s.flatten()
        #print("Array s is ", sArr[0:1])
        s_reduit = np.array([sArr[1], int(sArr[2]), int(sArr[3]), int(sArr[4] * 6 / math.pi)])
        #print("s_reduit=", s_reduit)
        return s_reduit

    def search_state(self, s_reduit, action):
        action = 8
        lignes = df[(df.robot_autonomous_current == s_reduit[0]) & (df.robot_x_current == s_reduit[1]) & (
                    df.robot_y_current == s_reduit[2]) & (df.robot_angle_current == s_reduit[3])]
        try:
            if (action == 8):
                state = lignes.sample(n=1).values[:, 15:102]
                #print("xxx=", lignes.values[1, 0:15])
            elif (action == 0):
                lignes = lignes[(lignes.man_auto == 1)]
                state = lignes.sample(n=1).values[:, 15:102]
            elif (action == 1):
                lignes = lignes[(lignes.auto_man == 1)]
                state = lignes.sample(n=1).values[:, 15:102]
            else:
                alarm = action - 2
                lignes = lignes.loc[lignes['alarm' + str(alarm)] == 1]
                #print("lignes trouve=", lignes)
                state = lignes.sample(n=1).values[:, 15:102]
        except ValueError:
            lignes = df[(df.robot_autonomous_current == s_reduit[0]) & (abs(df.robot_x_current-s_reduit[1]) <=1) & (
                    abs(df.robot_y_current - s_reduit[2])<=1) & (abs(df.robot_angle_current-s_reduit[3])<=1)]
            print("wow value error")
            state = lignes.sample(n=1).values[:, 15:102]

        print("new position=", state[:, 2:4],"auto or manuel mode=",state[:, 1:2])
        return state


NetToVerify = SavedNet()
TestAcc = []

str_filename = "dataHorizon/out/up_3.csv"

# Initial State
EPI_FILE = pd.read_csv(str_filename)
s_i = 3
s = np.array(EPI_FILE.values[s_i, 0:N_STATES])
# boucle
while (True):
    action = NetToVerify.choose_action(s)
    s_reduit = NetToVerify.state_transition(s)
    s_next = NetToVerify.search_state(s_reduit, action)
    s = s_next
    #print("new state=", s)
