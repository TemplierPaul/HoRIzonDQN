import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas as pd

# Hyper Parameters
EPI_FILE = pd.read_csv("dataHorizon/outNorm/up_0.csv")
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
N_NEURAL=32
N_EPISODE=660
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 32)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(32, 16)
        self.fc2.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(16, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

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
        #self.net = Net()

    def orig_action(self, x):
        action=np.array(EPI_FILE.ix[x, N_STATES :N_STATES+N_ACTIONS])
        action_index=8  # 8 means alarm 6, means None
        for i in range(N_ACTIONS):
            #print("i=",i)
            if(action[i]>0):
                action_index=i
                break
        return action_index

    def choose_action(self,s,orig_action):
        s = Variable(torch.FloatTensor(s))
        print("state=",s)
        q_next = self.net(s).detach()  # detach from graph, don't backpropagate/value of all the actions
        q_next=q_next.numpy()
        print("Orginal Action=",orig_action)
        print("Q-values=",q_next)
        print("Chosen Action=",np.argmax(q_next))
        return (np.argmax(q_next)==orig_action)

NetToVerify=SavedNet()
TestAcc=[]
for i_episode in range(660,1300):
    str_filename = "dataHorizon/outNorm/up_" + str(i_episode) + ".csv"
    try:
        EPI_FILE = pd.read_csv(str_filename)
    except FileNotFoundError:
        continue
    EPI_FILE = pd.read_csv(str_filename)

    N_EXP = EPI_FILE.iloc[:, 0].size
    s_i = 0
    s=np.array(EPI_FILE.ix[s_i, 0:N_STATES])

    ep_r = 0
    while (s_i<N_EXP-1):

        # take action
        orig_action = NetToVerify.orig_action(s_i)
        TestAcc.append(int(NetToVerify.choose_action(s,orig_action)))
        print("TestAccuarcy=",np.mean(TestAcc))

        r = np.array(EPI_FILE.ix[s_i, N_STATES+N_ACTIONS])
        s_i=s_i+1
        # if s_i>N_EXP_TOL:
        #     break
        s_next = np.array(EPI_FILE.ix[s_i, 0:N_STATES])
        print('Ep: ', i_episode)

        s = s_next
        #print(s)

