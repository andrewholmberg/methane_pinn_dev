import torch
import time
from torch import nn
from sklearn.mixture import GaussianMixture
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
class Gaussian_Mixture:

    def __init__(self,mean, st_dev, magnitude = None, trainable = True):
        mean = torch.tensor(mean)
        st_dev = torch.tensor(st_dev)
        magnitude = torch.tensor(magnitude)
        assert len(mean.shape)==2
        assert len(magnitude.shape) == 1
        for i in range(len(mean)-1):
            assert mean[i].shape == mean[i+1].shape
            assert st_dev[i].shape == mean[i].shape
            assert st_dev[i+1].shape == mean[i+1].shape

        self.mean = mean
        self.st_dev = st_dev
        self.spatial_dim = mean.shape[1]
        self.num_gaussian = len(mean)
        if trainable and magnitude is None:
            # self.magnitude = nn.Parameter(torch.rand(self.num_gaussian))
            self.magnitude =torch.rand(self.num_gaussian)

        elif not trainable and not magnitude is None:
            self.magnitude = magnitude
        elif trainable and not magnitude is None:
            # self.magnitude = nn.Parameter(magnitude)
            self.magnitude = torch.tensor(magnitude)
        else:
            self.magnitude = torch.rand(self.num_gaussian)

    def evaluate_wip(self,x):
        n = len(x)
        assert x.shape[1] == self.spatial_dim
        x_tiled = torch.tile(x,(self.num_gaussian,1))
        assert x_tiled.shape == (len(x)*self.num_gaussian,self.spatial_dim)
        source_pts = self.mean.repeat((len(x),1))
        assert x_tiled.shape == (len(x)*self.num_gaussian,self.spatial_dim)

        source_var = self.st_dev.repeat((len(x),1))
        assert source_var.shape == (len(x)*self.num_gaussian,self.spatial_dim)
        res = self.magnitude.repeat((len(x)))*1/(((2*torch.pi)**(self.spatial_dim/2))*torch.prod(source_var,dim=1))*torch.exp(-torch.sum(torch.square(x_tiled - source_pts)/(2*source_var**2),dim=1))
        assert res.view(-1,1).shape == (len(x)*self.num_gaussian,1)
        idx = torch.arange(0,len(res))
        filter = self.num_gaussian * (idx % n) + idx//n
        grouped_tensor = res[filter].view(-1,self.num_gaussian,len(x)).sum(dim=1).view(-1,1)

        assert grouped_tensor.shape == (len(x),1)
        return grouped_tensor
    
    def evaluate(self,x):
        n = len(x)
        assert x.shape[1] == self.spatial_dim
        base = torch.zeros(n,1)
        # print(self.magnitude)
        for i in range(self.num_gaussian):
            source_pts = self.mean[i].view(1,-1).repeat((len(x),1))
            source_var = self.st_dev[i].view(1,-1).repeat((len(x),1))
            assert source_pts.shape == x.shape
            # assert source_var.shape == (len(x)*self.num_gaussian,self.spatial_dim)
            assert source_var.shape == source_pts.shape
            assert source_var.shape == (n,self.spatial_dim)
            # res = self.magnitude[i]*1/(((2*torch.pi)**(self.spatial_dim/2))*torch.prod(source_var,dim=1))*torch.exp(-torch.sum(torch.square(x - source_pts)/(2*source_var**2),dim=1))
            res = self.magnitude[i]*1/(((2*torch.pi)**(self.spatial_dim/2))*torch.prod(source_var,dim=1))*torch.exp(torch.sum(-(x - source_pts)**2/(2*source_var**2),dim=1))
            base += res.view(-1,1)
        return base         
    '''
    def evaluate(self,x):
        assert x.shape[1] == self.spatial_dim

        # Correctly tile x along the rows
        x_tiled = x.repeat((self.num_gaussian, 1))
        assert x_tiled.shape == (len(x) * self.num_gaussian, self.spatial_dim)

        # Correctly repeat the means and variances
        source_pts = self.mean.repeat_interleave(len(x), dim=0)
        source_var = self.st_dev.repeat_interleave(len(x), dim=0)

        assert source_pts.shape == x_tiled.shape
        assert source_var.shape == x_tiled.shape

        # Compute the Gaussian PDF component-wise
        res = self.magnitude.repeat(len(x)) / (
            (2 * torch.pi) ** (self.spatial_dim / 2) * torch.prod(source_var, dim=1)
        ) * torch.exp(-torch.sum(torch.square(x_tiled - source_pts) / (2 * source_var**2), dim=1))

        # Reshape to match the grouping
        res = res.view(len(x), self.num_gaussian)

        # Sum the Gaussian contributions
        grouped_tensor = res.sum(dim=1).view(-1, 1)

        assert grouped_tensor.shape == (len(x), 1)
        return grouped_tensor
    '''


# gm = Gaussian_Mixture([[0,0,0],[1,1,1],[2,2,2]],[[1,1,1],[1,1,1],[1,1,1]])
# tensor = torch.rand(3,3)
# bt = time.time()
# print(gm.compute(tensor))
# et = time.time()
# print(et - bt)
# mean = [[0,0,0],[0,0,0],[100,100,100],[100,100,100],[100,100,100],[100,100,100],[100,100,100]]
# var = [[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]]
# magnitude = [1,1,1,1,1,1,1]



mean = np.array([[0,0,0],[0,0,0]])
var = [[.025,.025,.025],[.025,.025,.025]]
magnitude = [1,1]
gm = Gaussian_Mixture(mean,var,magnitude,trainable=False)
tensor = torch.rand(2,3)*.025
tensor = torch.rand(2,3)*.025
tensor = torch.tensor([[.01,.01,.01]])
bt = time.time()
x=gm.evaluate(tensor)
et = time.time()
print(et - bt)
print(x)


sgm = GaussianMixture(len(mean),covariance_type='full')
non_zero_idx = np.array(magnitude) > 0
sgm.weights_=np.array(magnitude)[non_zero_idx]
sgm.means_ = np.array(mean)[non_zero_idx]
sgm.covariances_ = np.array([np.eye(3)*.025 for _ in range(len(mean))])[non_zero_idx]
sgm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(sgm.covariances_))
source_term = torch.tensor(np.exp(sgm.score_samples(tensor.detach().cpu().numpy()))).view(tensor.shape[0],1)
print(source_term)
print(torch.mean(torch.abs(source_term - x)))
# print(torch.tensor([1,2,3,4])*torch.tensor([1,2,3,4]))

# print(torch.prod(torch.tensor([[1,2,3,4],[2,3,4,5]]),dim=1))

# res = torch.arange(0,12).view(4,3)
# print(res)
# n = 2
# idx = torch.arange(0,len(res))
# filter = n*(idx % n) + idx//n
# print(filter)
# print(res[filter].view(-1,3,n))
# print(res[filter].view(-1,3,n).sum(dim=1).sum(dim=1))

mean = [[0,0,0]]
stdev = [[.025,.025,.025]]
magnitude = [1]

mod = Gaussian_Mixture(mean,stdev,magnitude,False)
print(mod.evaluate(torch.tensor([[.01,.01,.01]])))