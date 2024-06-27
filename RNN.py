import torch
# import torchvision  # torch package for vision related things
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
from torch import nn  # All neural network modules
import random

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
# one row of the MNIST image is 28 pixels
# sequence_length = 28


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # input_size is the number of features for each time-step
        # we dont need to say explicitly how many sequences we want to have
        # RNN will work for any number of sequences we send (28 for MNIST)
        # batch_first use batch as the first axis --> (N, time_seq, features)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # hidden_size * sequence_length concatenates all the sequences from every hidden state
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        # x.size(0) is the number of mini-batches we send in at one time
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.rnn(x, h0)
        # keep the batch as the 1st axis, and concatenate everything else
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


# Recurrent neural network with GRU (many-to-one)
class RNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # change basic RNN to GRU
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate GRU
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


# Recurrent neural network with LSTM (many-to-one)
class RNN_LSTM_catAllHidden(nn.Module):
    """LSTM version that uses information from every hidden state"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM_catAllHidden, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # change basic RNN to LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # LSTM needs a separate cell state (LSTM needs both hidden and cell state)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        # need to give LSTM both hidden and cell state (h0, c0)
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out



class RNN_LSTM_Sophisticated(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM_Sophisticated, self).__init__()
        self.hidden_size = hidden_size 
        self.num_layers = num_layers + 2   
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))

        #attention
        attn_weights = F.softmax(self.attention(out).squeeze(-1), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), out).squeeze(1)
        out = self.fc(context)
        return out
    
    def sample_action(self, obs, epsilon):
        coin = random.random()
        if coin < epsilon():
            explore_action = random.randint(0, 4)
            return explore_action
        else:
            out = self.forward(obs)
            return out.argmax().item()





class RNN_LSTM_onlyLastHidden(nn.Module):
    """
    LSTM version that just uses the information from the last hidden state
    since the last hidden state has information from all previous states
    basis for BiDirectional LSTM
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM_onlyLastHidden, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # change basic RNN to LSTM
        # num_layers Default: 1
        # bias Default: True
        # batch_first Default: False
        # dropout Default: 0
        # bidirectional Default: False
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # remove the sequence_length
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Get data to cuda if possible
        x = x.to(device)
        # print("input x.size() = ", x.size())
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # LSTM needs a separate cell state (LSTM needs both hidden and cell state)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        # need to give LSTM both hidden and cell state (h0, c0)
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        # no need to reshape the out or concat
        # out is going to take all mini-batches at the same time + last layer + all features
        out = self.fc(out[:, -1, :])
        # print("forward out = ", out)
        return out

    def sample_action(self, obs, epsilon):
        # print("Sample Action called+++")
        """
        greedy epsilon choose
        """
        coin = random.random()
        if coin < epsilon:
            # print("coin < epsilon", coin, epsilon)
            # for 3actionStateEnv use [0,1,2]
            explore_action = random.randint(0,4)
            # print("explore_action = ", explore_action)
            return explore_action
        else:
            # print("exploit")
            out = self.forward(obs)
            return out.argmax().item()

# Create a bidirectional LSTM
class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # bidrectional=True for BiLSTM
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        # hidden_size needs to expand both directions, *2
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Get data to cuda if possible
        x = x.to(device)
        # print("input x.size() = ", x.size())
        # concat both directions, so need to times 2
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        # the _ is the (hidden_state, cell_state), but not used
        out, _ = self.lstm(x, (h0, c0))
        # only take the last hidden state to send to the linear layer
        out = self.fc(out[:, -1, :])

        return out

    def sample_action(self, obs, epsilon):
        # print("Sample Action called+++")
        """
        greedy epsilon choose
        """
        coin = random.random()
        if coin < epsilon:
            # print("coin < epsilon", coin, epsilon)
            # for 3actionStateEnv use [0,1,2]
            explore_action = random.randint(0,2)
            # print("explore_action = ", explore_action)
            return explore_action
        else:
            # print("exploit")
            out = self.forward(obs)
            return out.argmax().item()
