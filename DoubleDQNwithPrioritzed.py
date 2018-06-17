import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

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
MEMORY_CAPACITY = 500
N_EPISODE=660   #Number of files read (number of experiments)
N_EXP_TOL=400    #If the game is running too long, go to the next experiment(temporarily not considered)
N_ITERATION=20
N_NEURAL=32

class SumTree(object):
    """
    based on https://github.com/jaara/AI-blog/blob/master/SumTree.py
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root

class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    based on https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        abs_errors=abs_errors.detach()
        clipped_errors = np.minimum(np.array(abs_errors), self.abs_err_upper)
        #print("clipped_errors=",clipped_errors)
        ps = np.power(clipped_errors, self.alpha)
        #print("ps=",ps)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, N_NEURAL)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(N_NEURAL, N_NEURAL)
        self.fc2.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(N_NEURAL, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        # for target updating
        self.memory = Memory(capacity=MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.cost = []

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
        self.memory.store(transition)
        self.memory_counter += 1


    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        tree_idx,b_memory,ISWeights=self.memory.sample(BATCH_SIZE)
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 1 + 1]))
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
        loss4backward = self.loss_func(q_eval, q_target)
        #print("loss4backward=",type(loss4backward),loss4backward)
        #self.cost.append(loss4backward.detach().numpy())
        ISWeights=np.mean(ISWeights, axis=1)
        #print("ISWeights=",type(ISWeights),ISWeights)
        ISWeights = torch.from_numpy(ISWeights)
        ISWeights=ISWeights.type(torch.FloatTensor)
        loss=loss4backward*ISWeights
        self.cost.append(loss.detach().numpy())
        #print("loss=", type(loss), loss)
        abs_errors=torch.sum(torch.abs(q_target-q_eval),dim=1)
        #print("abs_errors=",abs_errors)
        self.memory.batch_update(tree_idx,abs_errors)


        self.optimizer.zero_grad()
        loss4backward.backward()
        self.optimizer.step()

dqn = DQN()
costs=[]
print('\nCollecting experience...')

for i in range(0, N_ITERATION):
    for i_episode in range(N_EPISODE):
        str_filename="dataHorizon/outNorm/up_"+str(i_episode)+".csv"
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

            r = np.array(EPI_FILE.ix[s_i, N_STATES+N_ACTIONS])
            s_i=s_i+1
            # if s_i>N_EXP_TOL:
            #     break
            s_next = np.array(EPI_FILE.ix[s_i, 0:N_STATES])


            dqn.store_transition(s, a_index, r, s_next)

            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                print('Ep: ', i_episode,
                    '| Ep_r: ', r)
                #print("weight=",dqn.target_net.fc1.weight)

            s = s_next
    costs.append(np.mean(dqn.cost))
    print(costs)


torch.save(dqn.eval_net, 'SavedNetwork/'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'DoubleDQNwithPrior_eval_net.pkl')
torch.save(dqn.target_net, 'SavedNetwork/'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'DoubbleDQNwithPrior_target_net.pkl')
log = open('Log/log'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'DoubleDQNPrio.txt', 'w')
log.write("cost value="+str(costs))
log.close()
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations')
plt.show()
