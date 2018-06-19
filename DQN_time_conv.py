import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

T_START = time.time()
# Hyper Parameters
EPI_FILE = pd.read_csv("dataHorizon/out/up_0.csv")
N_ACTIONS = 10
N_STATES = EPI_FILE.columns.size - N_ACTIONS - 1
print("N_ACTIONS:", N_ACTIONS)
print("N_STATES:", N_STATES)
BATCH_SIZE = 32   # Number of samples selected per study
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 10   # target update frequency
MEMORY_CAPACITY = 100
N_EPISODE=2   #Number of files read (number of experiments)
N_EXP_TOL=400    #If the game is running too long, go to the next experiment(temporarily not considered)
N_ITERATION=2
N_NEURAL=32

N_PAST_STATES = 10 #number of states taken for history in convolution
N_CONV_OUT = N_STATES
N_LINEAR_IN = 400

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv_1", torch.nn.Conv2d(1, 100, kernel_size=3))
        self.conv.add_module("maxpool_1", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("relu_1", torch.nn.ReLU())
        self.conv.add_module("conv_2", torch.nn.Conv2d(100, 20, kernel_size=3))
        self.conv.add_module("dropout_2", torch.nn.Dropout())
        self.conv.add_module("maxpool_2", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("relu_2", torch.nn.ReLU())

        self.fc = torch.nn.Sequential()
        self.fc.add_module("fc1", torch.nn.Linear(N_LINEAR_IN, 50))
        self.fc.add_module("relu_3", torch.nn.ReLU())
        self.fc.add_module("dropout_3", torch.nn.Dropout())
        self.fc.add_module("fc2", torch.nn.Linear(50, N_CONV_OUT))

        self.fc1 = nn.Linear(N_STATES + N_CONV_OUT, N_NEURAL)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(N_NEURAL, N_NEURAL)
        self.fc2.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(N_NEURAL, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x, prev):
        # print ("Forward : x, prev = ")
        # print (x.shape)
        # print(prev.shape)

        prev = self.conv.forward(prev)
        prev = prev.view(-1, N_LINEAR_IN)
        prev = self.fc.forward(prev)

        # print ("Apres CNN : x, prev")
        # print(x.shape)
        # print(prev.shape)
        combined = torch.cat((x, prev),1)
        # print(combined.shape)
        combined = self.fc1(combined)
        combined = F.relu(combined)
        combined = self.fc2(combined)
        combined = F.relu(combined)
        actions_value = self.out(combined)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 1 + 1))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.cost = []

    def choose_action(self, x):
        action=np.array(EPI_FILE.ix[x, N_STATES :N_STATES+N_ACTIONS])
        action_index=8  # 8 means alarm 6, means None
        for i in range(N_ACTIONS):
            #print("i=",i)
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
        sample_index = np.random.choice(range(N_PAST_STATES,MEMORY_CAPACITY), BATCH_SIZE)
        # print("Sample index : ")
        # print(sample_index)
        b_memory = self.memory[sample_index , :]
        mem = []
        for i in sample_index:
            if (i >= N_PAST_STATES):
                mem.append(self.memory[i - N_PAST_STATES : i, :N_STATES])
                #print (str(i) + ' < N_PAST_STATES ')
                #print(mem[-1].shape)
            else :
                mem.append([])
                for j in range (N_PAST_STATES - i):
                    mem [-1] = np.concatenate(mem[-1], self.memory[0, :N_STATES])
                mem [-1] =  np.concatenate(mem[-1], self.memory[0 : i, :N_STATES])
                print (str(i) + ' > N_PAST_STATES ')
                print (mem[-1].shape)
        b_prev_s =  Variable(torch.FloatTensor(mem))
        # print (b_prev_s.shape)
        b_prev_s = torch.unsqueeze(b_prev_s, 1)
        # print(b_prev_s.shape)
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        # print(b_s.shape)
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)))
        #print("b_a=",b_a)
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+1+1]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))
        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s, b_prev_s).gather(1, b_a )# shape (batch, 1)//value of the chosen action
        q_next = self.target_net(b_s_, b_prev_s).detach()     # detach from graph, don't backpropagate/value of all the actions
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        #print("b_a=",b_a)
        #print("q_eval=",q_eval)
        #print("q_target=",q_target)
        loss = self.loss_func(q_eval, q_target)
        self.cost.append(loss.detach().numpy())
        #print("loss=",loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()
costs=[]
print('\nCollecting experience...')
for i in range(0, N_ITERATION):
    dqn.cost=[]
    for i_episode in range(N_EPISODE):
        str_filename="dataHorizon/out/up_"+str(i_episode)+".csv"
        try:
            EPI_FILE = pd.read_csv(str_filename)
        except FileNotFoundError:
            continue
        N_EXP = EPI_FILE.iloc[:, 0].size
        s_i = 0
        s=np.array(EPI_FILE.ix[s_i, 0:N_STATES])
        while (s_i<N_EXP-1):

            # take action
            a_index = dqn.choose_action(s_i)
            #print("a_index=",a_index)
            r = np.array(EPI_FILE.ix[s_i, N_STATES+N_ACTIONS])
            s_i=s_i+1
            # if s_i>N_EXP_TOL:
            #     break
            s_next = np.array(EPI_FILE.ix[s_i, 0:N_STATES])


            dqn.store_transition(s, a_index, r, s_next)


            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                print("cost=",np.sum(dqn.cost))
                print('Ep: ', i_episode,
                    '| Ep_r: ', r, '|Time: ', time.time()-T_START)
                # print("weight=",dqn.target_net.fc1.weight)

            s = s_next

    costs.append(np.mean(dqn.cost))
    print(costs)


torch.save(dqn.eval_net, 'SavedNetwork/'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'DQN_eval_net.pkl')
torch.save(dqn.target_net, 'SavedNetwork/'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'DQN_target_net.pkl')
log = open('Log/log'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'DQN.txt', 'w')
log.write("cost value="+str(costs))
log.close()
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations')
plt.show()
