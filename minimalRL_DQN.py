from collections import deque

import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gamma = 0.98
batch_size = 32
train_times = 10

class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = deque(maxlen = buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)
        
        s_lst = np.array(s_lst)
        a_lst = np.array(a_lst)
        r_lst = np.array(r_lst)
        s_prime_lst = np.array(s_prime_lst)
        done_mask_lst = np.array(done_mask_lst)

        return torch.tensor(s_lst, device=device, dtype=torch.float), torch.tensor(a_lst, device=device, dtype=torch.float), \
               torch.tensor(r_lst, device=device, dtype=torch.float), torch.tensor(s_prime_lst, device=device, dtype=torch.float), \
               torch.tensor(done_mask_lst, device=device, dtype=torch.float)
    
    def size(self):
        return len(self.buffer)
    
    def save(self, save_path):
        with open(save_path, "wb") as handle:
            self.buffer = pickle.load(handle)
    
    def load(self, file_path):
        with open(file_path, "rb") as handle:
            self.buffer = pickle.load(handle)
        
#reservoir layer

class ReservoirLayer(nn.Module):
    def __init__(self, input_size, reservoir_size, spectral_radius = 1.1):
        super(ReservoirLayer, self).__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius

        self.input_weights = nn.Parameter(torch.Tensor(reservoir_size, input_size), requires_grad = False)
        self.reservoir_weights = nn.Parameter(torch.Tensor(reservoir_size, reservoir_size), requires_grad = False)

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.input_weights, mean = 0, std = 1)
        nn.init.normal_(self.reservoir_weights, mean = 0, std = 1)
        radius = np.max(np.abs(np.linalg.eigvals(self.reservoir_weights.detach().numpy())))
        self.reservoir_weights.data *= self.spectral_radius / radius
    
    def forward(self, x):
        batch_size, input_size = x.size()
        state = torch.zeros(batch_size, self.reservoir_size, device = x.device)

        for i in range(input_size // self.input_size):
            input_slice = x[:, i*self.input_size:(i+1)*self.input_size]
            state = torch.tanh(torch.matmul(input_slice, self.input_weights.t()) + torch.matmul(state, self.reservoir_weights))
        
        return state

#linear layer

class FCN_QNet(nn.Module):
    def __init__(self, insize, outsize, reservoir_size= 300):
        super(FCN_QNet, self).__init__()
        self.insize = insize
        self.reservoir = ReservoirLayer(insize, reservoir_size)
        self.fc1 = nn.Linear(reservoir_size, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 64)
        self.fc4 = nn.Linear(64, outsize)
    
    def forward(self, x):
        x = x.to(device)
        x = x.view(x.size(0), -1)
        x = self.reservoir(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x

    def sample_action(self, obs, epsilon):
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 4)
        else:
            out = self.forward(obs)
            return out.argmax(dim = -1).item()

def train(q, q_target, memory, optimizer):
    loss_values = []
    for i in range(train_times):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        q_out = q(s)
        a = a.reshape(a.size(0), 1)  # Reshape a to have the same number of dimensions as q_out
        q_a = q_out.gather(1, a.long())
        max_q_prime = q_target(s_prime).max(1)[0]  # Remove the unsqueeze operation
        target = r + gamma * max_q_prime * done_mask
        loss = F.mse_loss(q_a, target.unsqueeze(1))  # Unsqueeze target to match the shape of q_a

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())  # Append the loss value to the list

    return loss_values

