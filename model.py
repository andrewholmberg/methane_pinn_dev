import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
# create a multilayer perceptron
class Net(torch.nn.Module):
    def __init__(self, spatial_dim, hidden_size, output_size):
        super(Net, self).__init__()
        self.spatial_dim = spatial_dim
        self.hidden = torch.nn.Linear(spatial_dim+1, hidden_size)
        self.hidden2 = torch.nn.Linear(hidden_size, hidden_size)
        self.hidden3 = torch.nn.Linear(hidden_size, hidden_size)

        # self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.output = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x,t):
        # assumes x has shape (batch_size, spatial_dim)
        # assumes t has shape (batch_size, 1)

        # print('x.shape =    ', x.shape)
        # print('t.shape = ', t.shape)

        # concatenate x and t
        xt = torch.cat([x,t], dim=1)

        xt = self.tanh(self.hidden(xt))
        xt = self.tanh(self.hidden2(xt))
        xt = xt + self.tanh(self.hidden3(xt))
        xt = self.output(xt)

        # ensures at t=0, the output is zero
        xt = t*xt

        return xt
    
    # create loss for  \|u_t - u_xx = 0\|

    def loss_fn(model, x, t, v, source_loc, source_value):
        # assumes v has shape (1, spatial_dim)
        # assumes source_loc has shape (1, spatial_dim)
        # assumes source_value is a scalar

        batch_size = x.shape[0]
        spatial_dim = x.shape[1]
        # print('x.shape = ', x.shape)
        # print('t.shape = ', t.shape)
        assert t.shape[0] == batch_size

        # compute gradients
        x.requires_grad = True
        t.requires_grad = True
        u = model(x,t)
        u_x = torch.autograd.grad(outputs=u, inputs=x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True, allow_unused=True)[0]
        u_xx = torch.autograd.grad(outputs=u_x, inputs=x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True, allow_unused=True)[0]
        u_t = torch.autograd.grad(outputs=u, inputs=t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

        assert u_x.shape == (batch_size, spatial_dim)
        assert u_xx.shape == (batch_size, spatial_dim)
        assert u_t.shape == (batch_size ,1)

        laplace_term = torch.sum(u_xx, dim=1).view(batch_size, 1)
        velocity_term = torch.sum(v*u_x[:,:2], dim=1).view(batch_size, 1)
        # print(u_x[:,:2])
        assert laplace_term.shape == (batch_size, 1)
        assert velocity_term.shape == (batch_size, 1)
        assert u_t.shape == (batch_size, 1)

        # source_term  = source_value*torch.prod(x == source_loc, dim=1).view(-1, 1)

        sigma = 0.05  # Small value for steep Gaussian

        # Compute pairwise squared Euclidean distances
        diff = x.unsqueeze(1) - source_loc.unsqueeze(0)  # Shape: (n, m, d)
        squared_distances = torch.sum(diff**2, dim=2)    # Shape: (n, m)

        # Compute Gaussian source term
        source_term = source_value * torch.exp(-squared_distances / (2 * sigma**2))  # Shape: (n, m)

        t0 = torch.zeros_like(t)
        u0 = model(x, t0)

        init_loss = torch.mean(u0**2)
        negative_loss = torch.mean((torch.abs(u) - u)**2)
        # compute loss
        pde_loss = torch.mean((u_t + velocity_term -  laplace_term - source_term)**2) 

        total_loss = pde_loss + init_loss + negative_loss

        return total_loss, pde_loss, init_loss