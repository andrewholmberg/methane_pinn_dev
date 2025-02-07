import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
class Net(torch.nn.Module):
    def __init__(self, spatial_dim, hidden_size):
        super(Net, self).__init__()
        output_size = 1
        hidden_size= 100
        self.spatial_dim = spatial_dim
        self.hidden = torch.nn.Linear(spatial_dim+1, hidden_size)
        self.hidden2 = torch.nn.Linear(hidden_size, hidden_size)
        self.hidden3 = torch.nn.Linear(hidden_size, hidden_size)

        # self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.output = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # assumes x has shape (batch_size, spatial_dim)
        # assumes t has shape (batch_size, 1)

        # print('x.shape =    ', x.shape)
        # print('t.shape = ', t.shape)

        # concatenate x and t
        # xt = torch.cat([x,t], dim=1)

        t = x[:,0:1]
        xt = self.tanh(self.hidden(x))
        xt = self.tanh(self.hidden2(xt))
        xt = xt + self.tanh(self.hidden3(xt))
        xt = self.output(xt)

        # ensures at t=0, the output is zero
        xt = t*xt
        print(xt.shape)
        return xt