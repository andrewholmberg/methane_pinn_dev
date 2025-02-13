from Net import Net
# from Net_1 import Net
import torch
import numpy as np
import pandas as pd
from Gaussian_Mixture import Gaussian_Mixture
from sklearn.mixture import GaussianMixture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
class PINN:
    def __init__(self,hidden_structure):
        self.spatial_dim = 3
        self.net = Net(self.spatial_dim, hidden_structure)
        self.collocation_points = None
        self.initial_condition_points = None
        self.l_scale = 1
        self.t_scale = 1
    
    def set_location(self, source_locs, max_vals, source_values = None, kappa = 1e-2, sigma = .025, trainable = False):
        self.source_locs = source_locs
        #figure out if there should be 3 individual scales, or just 1 across x,y,z
        self.l_scale = 1 #max(max_vals[1:])
        self.source_locs_scaled = source_locs / self.l_scale
        self.t_scale = 1 #max_vals[0]
        self.t_max = max_vals[0]

        self.source_mixture_hm = Gaussian_Mixture(source_locs,[[sigma]*self.spatial_dim for _ in range(len(source_values))],source_values,trainable)
        self.q = self.source_mixture_hm.magnitude
        if source_values is not None:
            assert len(source_values) == len(source_values)
            self.q = source_values
            self.source_mixture = GaussianMixture(len(source_locs),covariance_type='full')
            non_zero_idx = np.array(source_values) > 0
            self.source_mixture.weights_=np.array(source_values)[non_zero_idx]
            self.source_mixture.means_ = np.array(source_locs)[non_zero_idx]
            self.source_mixture.covariances_ = np.array([np.eye(3)*sigma for _ in range(len(source_values))])[non_zero_idx]
            self.source_mixture.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(self.source_mixture.covariances_))
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

    def scale_tensor(self, loc_tensor, wind_tensor = None):
        loc_temp = loc_tensor.clone()
        loc_temp[:,1:] = loc_temp[:,1:]/self.l_scale
        loc_temp[:,0] = loc_temp[:,0]/self.t_scale
        if wind_tensor != None:
            wind_temp = wind_tensor.clone()*self.t_scale/self.l_scale
        return loc_temp, wind_temp


    def source_points(self,n,sigma):
        torch.empty(0,4)
        source_inputs_ls = torch.empty(0,4)
        # uv_inputs_ls = torch.empty(0,2)
        for i in range(len(self.q)):
            
            if self.q[i] > .001:
                rand_source = torch.tensor(np.tile(self.source_locs[i],(n,1)) + np.random.randn(n,3)*sigma)
                # Compute Gaussian source term
                rand_time = torch.rand(n,1)*self.t_max # Shape: (61,)

                source_inputs = torch.cat([rand_time,rand_source],dim=1)
                # Repeat space locations for each time step
                # Repeat time steps for each space location
                # Concatenate space locations and time
                # uv = torch.cat([torch.ones(len(source_inputs)).view(-1,1)*-1, torch.ones(len(source_inputs)).view(-1,1)*-1], dim=1)
                source_inputs_ls= torch.cat([source_inputs_ls, source_inputs])
                # uv_inputs_ls = torch.cat([uv_inputs_ls,uv])
        return source_inputs_ls.float()
    

    def compute_pde_loss(self, tx, wind_vector, source_term = None, scaled = False):
        # assumes v has shape (1, spatial_dim)
        # assumes source_loc has shape (1, spatial_dim)
        # assumes source_value is a scalar
        if scaled == False:
            tx, wind_vector = self.scale_tensor(tx, wind_vector)

        batch_size = tx.shape[0]
        spatial_dim =self.spatial_dim

        tx.requires_grad_()
        u = self.net(tx)
        u_x = torch.autograd.grad(outputs=u, inputs=tx, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True, allow_unused=True)[0]
        u_xx = torch.autograd.grad(outputs=u_x, inputs=tx, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True, allow_unused=True)[0]
        
        assert u.shape == (batch_size,1)

        # u_t = torch.autograd.grad(outputs=u, inputs=t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

        assert u_x.shape == (batch_size, spatial_dim+1)
        assert u_xx.shape == (batch_size, spatial_dim+1)

        laplace_term = torch.sum(u_xx[:,1:], dim=1).view(batch_size, 1)
        assert wind_vector.shape == u_x[:,1:3].shape
        velocity_term = torch.sum(wind_vector*u_x[:,1:3], dim=1).view(batch_size, 1)

        # print(u_x[:,:2])
        assert laplace_term.shape == (batch_size, 1)
        assert velocity_term.shape == (batch_size, 1)
        # assert u_t.shape == (batch_size, 1)
        source_term = self.source_mixture_hm.evaluate(tx[:,1:]).view(batch_size,1)
        # print(source_term)
        # source_term = torch.tensor(source_term.clone().detach().cpu().numpy())

        # source_term = torch.tensor(np.exp(self.source_mixture.score_samples(tx[:,1:].detach().cpu().numpy()))).view(batch_size,1)
        # print(self.source_mixture_hm.mean)
        # print(self.source_mixture_hm.st_dev)
        # print(self.source_mixture_hm.magnitude)
        # print(torch.cat([source_term_hm,source_term],dim=1))
        # print( source_term[0,0] , source_term_hm[0,0])
        # assert source_term.shape == source_term_hm.shape
        # print(tx[:,1:],source_term)
        # source_term = torch.tensor(source_term.detach().cpu().numpy())
        assert source_term.shape == (batch_size, 1)
        # print(source_term)
        # compute loss
        assert u_x[:,0:1].shape == velocity_term.shape
        kappa = 1*1e-3
        # print(source_term)
        assert u_x[:,0:1].shape ==velocity_term.shape == laplace_term.shape == source_term.shape
        # print(tx[-2])
        pde_loss = torch.mean( torch.square((u_x[:,0:1] + velocity_term - kappa * laplace_term - source_term) ))

        total_loss = pde_loss
        return total_loss, pde_loss 
    def compute_negative_loss(self,points):
        u = self.forward(points)
        return torch.mean((torch.abs(u)-u)**2)
    
    def compute_data_loss(self,data,data_values):

        assert data.shape == (data_values.shape[0],self.spatial_dim+1)
        assert len(data.shape) == 2
        assert data_values.shape[1] == 1
        return torch.mean(torch.square(self.forward(data,False) - data_values))

    def train(self,num_epochs, initial_condition = None, collocation = None):
        if initial_condition == None:
            initial_condition = self.initial_condition_points
        if collocation == None:
            collocation = self.collocation_points
        
