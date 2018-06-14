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
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 10   # target update frequency
MEMORY_CAPACITY = 1000
N_EPISODE=100
N_EXP_TOL=200
N_HIDDEN = 100



class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        # input to hidden
        self.fc1h = nn.Linear(input_size + hidden_size, 50)
        self.fc1h.weight.data.normal_(0, 0.1)   # initialization
        self.outh = nn.Linear(50, hidden_size)
         # input to output
        self.fc1o = nn.Linear(input_size + hidden_size, 50)
        self.fc1o.weight.data.normal_(0, 0.1)   # initialization
        self.outo = nn.Linear(50, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        # hidden
        hidden = self.fc1h(combined)
        hidden = F.relu(hidden)
        hidden = self.outh(hidden)
        #output
        output = self.fc1o(combined)
        output = F.relu(output)
        output = self.outo(output)

        output = self.softmax(output)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class DRQN(object):
    def __init__(self):
        self.eval_net, self.target_net = RNN(N_STATES, N_HIDDEN, N_ACTIONS), RNN(N_STATES, N_HIDDEN, N_ACTIONS)

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 1 + 1))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        action=np.array(EPI_FILE.ix[x, N_STATES :N_STATES+N_ACTIONS])
        action_index=0
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
        #sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE) #TOMODIFY
        b_memory = self.memory[0:BATCH_SIZE, :]#[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES])) #state
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))) # action
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+1+1])) # reward
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:])) # next state
        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a )# shape (batch, 1)//value of the chosen action
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate/value of all the actions
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        #print("b_a=",b_a)
        #print("q_eval=",q_eval)
        #print("q_target=",q_target)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


criterion = nn.NLLLoss()

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-LR, p.grad.data)

    return output, loss.item()


dqn = DRQN()

print('\nCollecting experience...')
for i_episode in range(N_EPISODE):
    str_filename="dataHorizon/out/up_"+str(i_episode)+".csv"
    try:
        EPI_FILE = pd.read_csv(str_filename)
    except FileNotFoundError:
        continue
    N_EXP = EPI_FILE.iloc[:, 0].size
    s_i = 1
    s=np.array(EPI_FILE.ix[s_i, 0:N_STATES])
    ep_r = 0
    while (s_i<N_EXP-1):
        # take action
        a_index = dqn.choose_action(s_i)

        r = np.array(EPI_FILE.ix[s_i, N_STATES+N_ACTIONS])
        s_i=s_i+1
        if s_i>N_EXP_TOL:
            break
        s_next = np.array(EPI_FILE.ix[s_i, 0:N_STATES])

        dqn.store_transition(s, a_index, r, s_next)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            print('Ep: ', i_episode,
                '| Ep_r: ', round(ep_r, 2))
            #print("weight=",dqn.target_net.fc1.weight)
        s = s_next

torch.save(dqn.eval_net, 'DRQN_eval_net.pkl')
torch.save(dqn.target_net, 'DRQN_target_net.pkl')
