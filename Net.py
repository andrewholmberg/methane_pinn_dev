import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
# create a multilayer perceptron
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
class Net(torch.nn.Module):

    def __init__(self,spatial_dim, hidden_structure:list):
        super(Net, self).__init__()
        assert len(hidden_structure)>=2

        self.spatial_dim = spatial_dim
        self.hidden_structure= hidden_structure
        self.hidden = torch.nn.ModuleList()        
        self.hidden.append(torch.nn.Linear(spatial_dim+1,hidden_structure[0]))
        for i in range(1,len(hidden_structure)):
            self.hidden.append(torch.nn.Linear(hidden_structure[i-1], hidden_structure[i]))
        self.hidden.append(torch.nn.Linear(hidden_structure[-1],1))
        self.apply(init_weights)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.1)

    def forward(self, input_tensor):
        #t is the last column in the input tensor
        t = input_tensor[:,0:1]
    
        xt = self.leaky_relu(self.hidden[0](input_tensor))
        for i in range(1,len(self.hidden)-2):
            xt = xt + self.leaky_relu(self.hidden[i](xt))
        xt = xt + self.tanh(self.hidden[-2](xt))
        xt = self.hidden[-1](xt)
        # ensures at t=0, the output is zero
        # ensures at t=0, the output is zero
        xt = t*xt
        return xt
    '''
    # create loss for  \|u_t - u_xx = 0\|

class Net(torch.nn.Module):
    def __init__(self, spatial_dim, hidden_size ):
        super(Net, self).__init__()
        self.spatial_dim = spatial_dim
        self.hidden = torch.nn.Linear(spatial_dim+1, hidden_size)
        self.hidden2 = torch.nn.Linear(hidden_size, hidden_size)
        self.hidden3 = torch.nn.Linear(hidden_size, hidden_size)

        # self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.output = torch.nn.Linear(hidden_size, 1)

    def forward(self, xt):
        print(xt)
        # assumes x has shape (batch_size, spatial_dim)
        # assumes t has shape (batch_size, 1)

        # print('x.shape =    ', x.shape)
        # print('t.shape = ', t.shape)

        # concatenate x and t
        t = xt[:,0]

        xt = self.tanh(self.hidden(xt))
        xt = self.tanh(self.hidden2(xt))
        xt = xt + self.tanh(self.hidden3(xt))
        xt = self.output(xt)

        # ensures at t=0, the output is zero
        xt = t*xt

        return xt
        '''