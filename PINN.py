from Net import Net
import torch
import numpy as np
import pandas as pd
class PINN:
    def __init__(self,hidden_structure):
        self.spatial_dim = 3
        self.net = Net(self.spatial_dim, hidden_structure)
        self.collocation_points = None
        self.initial_condition_points = None
        self.l_scale = 1
        self.t_scale = 1
    
    def set_location(self, source_locs, max_vals, source_values = None):
        self.source_locs = source_locs
        #figure out if there should be 3 individual scales, or just 1 across x,y,z
        self.l_scale = 1#max(max_vals[1:])
        self.source_locs_scaled = source_locs / self.l_scale
        self.t_scale = 1#max_vals[0]
        if source_values != None:
            assert len(source_values) == len(source_values)
            self.q = source_values
        else:
            self.q = [0] * len(source_locs)

    def set_default_collocation_points(self,collocation_points):
        self.collocation_points = collocation_points

    def forward(self,input_tensor,scaled=False):
        if scaled:
            return self.net(input_tensor)
        else:
            temp = input_tensor.clone()
            temp[:,1:] = temp[:,1:]/self.l_scale
            temp[:,0] = temp[:,0]/self.t_scale
            return self.net(temp)

    def scale_tensor(self,loc_tensor, wind_tensor = None):
        loc_temp = loc_tensor.clone()
        loc_temp[:,1:] = loc_temp[:,1:]/self.l_scale
        loc_temp[:,0] = loc_temp[:,0]/self.t_scale
        if wind_tensor != None:
            wind_temp = wind_tensor.clone()*self.t_scale/self.l_scale
        return loc_temp, wind_temp


    def source_points(self,n,sigma):
        torch.empty(0,4)
        source_inputs_ls = torch.empty(0,4)
        source_values_ls = torch.empty(0,1)
        uv_inputs_ls = torch.empty(0,2)
        for i in range(len(self.q)):
            
            if self.q[i] > .001:
                rand_source = torch.tensor(np.tile(self.source_locs[i],(n,1)) + np.random.randn(n,3)*sigma)
                source_stacked = torch.tensor(np.tile(self.source_locs[i],(n,1)))
                diff = rand_source - source_stacked  # Shape: (n, m, d)
                squared_distances = torch.sum(diff**2, dim=1)    # Shape: (n, m)
                # Compute Gaussian source term
                source_values = self.q[i] * torch.exp(-squared_distances / (2 * sigma**2)).float().view(-1,1)  # Shape: (n, m)
                rand_time = torch.rand(n,1)*self.t_scale  # Shape: (61,)
                source_inputs = torch.cat([rand_time,rand_source],dim=1)
                # Repeat space locations for each time step
                # Repeat time steps for each space location
                # Concatenate space locations and time
                uv = torch.cat([torch.ones(len(source_inputs)).view(-1,1)*.5, torch.ones(len(source_inputs)).view(-1,1)*.5], dim=1)
                source_inputs_ls= torch.cat([source_inputs_ls, source_inputs])
                source_values_ls =torch.cat([source_values_ls, source_values])
                uv_inputs_ls = torch.cat([uv_inputs_ls,uv])
        return source_inputs_ls.float(), uv_inputs_ls.float(), source_values_ls.float()
    

    '''
    def source_points(self,n,sigma):
        torch.empty(0,4)
        source_inputs_ls = torch.empty(0,4)
        source_values_ls = torch.empty(0,1)
        uv_inputs_ls = torch.empty(0,2)
        for i in range(len(self.q)):
            
            if self.q[i] > .001:
                rand_source = torch.tensor(np.tile(self.source_locs[i],(n,1)) + np.random.randn(n,3)*sigma)
                source_stacked = torch.tensor(np.tile(self.source_locs[i],(n,1)))
                diff = rand_source - source_stacked  # Shape: (n, m, d)
                squared_distances = torch.sum(diff**2, dim=1)    # Shape: (n, m)
                # Compute Gaussian source term
                source_values = self.q[i] * torch.exp(-squared_distances / (2 * sigma**2)).float().view(-1,1)  # Shape: (n, m)
                time_steps = torch.linspace(0, self.t_scale, steps=100)  # Shape: (61,)
                # Repeat space locations for each time step
                space_repeated = rand_source.repeat(len(time_steps), 1)  # Shape: (61 * n_locations, 3)
                source_values_repeated = source_values.repeat(len(time_steps),1)
                # Repeat time steps for each space location
                time_repeated = time_steps.repeat_interleave(rand_source.shape[0])  # Shape: (61 * n_locations,)
                # Concatenate space locations and time
                source_inputs = torch.cat((time_repeated.unsqueeze(1),space_repeated), dim=1).float()  # Shape: (61 * n_locations, 4)
                uv = torch.cat([torch.ones(len(source_inputs)).view(-1,1)*.5,torch.ones(len(source_inputs)).view(-1,1)*.5],dim=1)
                source_inputs_ls= torch.cat([source_inputs_ls, source_inputs])
                source_values_ls =torch.cat([source_values_ls,source_values_repeated])
                uv_inputs_ls = torch.cat([uv_inputs_ls,uv])
                pd.DataFrame(torch.cat([source_inputs_ls,source_values_ls],dim=1).detach().cpu().numpy()).to_csv('output.csv')
                raise Exception('STOP!!')
        return source_inputs_ls, uv_inputs_ls, source_values_ls
        '''
    def loss_function(self, tx, uv, source_term = None, scaled = False):
        # assumes v has shape (1, spatial_dim)
        # assumes source_loc has shape (1, spatial_dim)
        # assumes source_value is a scalar
        if scaled == False:
            tx, uv = self.scale_tensor(tx, uv)

        batch_size = tx.shape[0]
        spatial_dim =self.spatial_dim
        # print('x.shape = ', x.shape)
        # print('t.shape = ', t.shape)
        # assert t.shape[0] == batch_size

        # compute gradients
        # x.requires_grad = True
        # t.requires_grad = True
        tx.requires_grad_()
        u = self.net(tx)
        u_x = torch.autograd.grad(outputs=u, inputs=tx, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True, allow_unused=True)[0]
        u_xx = torch.autograd.grad(outputs=u_x, inputs=tx, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True, allow_unused=True)[0]
        
        assert u.shape == (batch_size,1)
        # u_t = torch.autograd.grad(outputs=u, inputs=t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

        assert u_x.shape == (batch_size, spatial_dim+1)
        assert u_xx.shape == (batch_size, spatial_dim+1)

        laplace_term = torch.sum(u_xx[:,1:], dim=1).view(batch_size, 1)
        velocity_term = torch.sum(.5*u_x[:,1:3], dim=1).view(batch_size, 1)

        # print(u_x[:,:2])
        assert laplace_term.shape == (batch_size, 1)
        assert velocity_term.shape == (batch_size, 1)
        # assert u_t.shape == (batch_size, 1)

        # source_term  = source_value*torch.prod(x == source_loc, dim=1).view(-1, 1)
        sigma = 0.05  # Small value for steep Gaussian
        # Compute pairwise squared Euclidean distances
        source_loc = torch.tensor(self.source_locs)
        diff = tx[:,1:].unsqueeze(1) - source_loc.unsqueeze(0)  # Shape: (n, m, d)
        squared_distances = torch.sum(diff**2, dim=2)    # Shape: (n, m)
        # Compute Gaussian source term
        source_term = 10 * torch.exp(-squared_distances / (2 * sigma**2))  # Shape: (n, m)

        negative_loss = torch.mean((torch.abs(u) - u)**2)
        # compute loss

        pde_loss = torch.mean((u_x[:,0] + velocity_term - 0.0 * laplace_term - (source_term if source_term != None else 0) )**2) 

        total_loss = pde_loss + negative_loss
        # print(torch.mean(velocity_term**2),torch.mean(source_term**2),torch.mean(u_x[:,0]**2))
        return total_loss, pde_loss, 
    
    def train(self,num_epochs, initial_condition = None, collocation = None):
        if initial_condition == None:
            initial_condition = self.initial_condition_points
        if collocation == None:
            collocation = self.collocation_points
        
