import torch
import time
from torch import nn
from sklearn.mixture import GaussianMixture
import numpy as np

torch.set_printoptions(precision=8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


'''
Class Gaussian_Mixture
description: approximate source delta function with a gaussian. Takes source locations (mean), standard deviations for the gaussians,
    and magnitudes to indicate how much a source is emitting.
'''
class Gaussian_Mixture:
    '''
    function __init__ - constructor.
    @param mean - the locations of the sources
    @param st_dev - the standard deviations of the sources
    @param magnitude - height / scale of the gaussians - higher value indicates more emission.
    @param trainable - whether or not the magnitudes are trainable. self.magnitude is param tensor if true, else normal tensor.
    @return - None
    '''
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
            self.magnitude = nn.Parameter(torch.rand(self.num_gaussian,requires_grad=True).float())

        elif not trainable and not magnitude is None:
            self.magnitude = magnitude.float()
        elif trainable and not magnitude is None:
            self.magnitude = nn.Parameter(torch.tensor(magnitude,requires_grad=True).float())
        else:
            self.magnitude = torch.rand(self.num_gaussian).float()


    '''
    function evaluate - evaluate the gaussian mixture at a given point(s). evaluate point at each of the gaussian
        distributions, then add them all up
    @param x - tensor of (n,3) points to evaluate the gaussians
    @return - toret tensor of (n,1) with the combined source values.
    '''
    def evaluate(self,x):
        n = len(x)
        assert x.shape[1] == self.spatial_dim
        base = torch.zeros(n,1)
        # print(self.magnitude)
        tensor = torch.zeros(n,self.num_gaussian).float()
        for i in range(self.num_gaussian):
            source_pts = self.mean[i].view(1,-1).repeat((len(x),1))
            source_stdev = self.st_dev[i].view(1,-1).repeat((len(x),1))
            assert source_pts.shape == x.shape
            # assert source_stdev.shape == (len(x)*self.num_gaussian,self.spatial_dim)
            assert source_stdev.shape == source_pts.shape
            assert source_stdev.shape == (n,self.spatial_dim)
            # res = self.magnitude[i]*1/(((2*torch.pi)**(self.spatial_dim/2))*torch.prod(source_stdev,dim=1))*torch.exp(-torch.sum(torch.square(x - source_pts)/(2*source_stdev**2),dim=1))
            res = 1/(((2*torch.pi)**(self.spatial_dim/2))*torch.prod(source_stdev,dim=1))*torch.exp(torch.sum(-(x - source_pts)**2/(2*source_stdev**2),dim=1))
            tensor[:,i] = res

        toret = (torch.clamp(self.magnitude,0,10) @ torch.transpose(tensor.float(),0,1)).view(-1,1)
        # print(torch.max(toret))
        return toret



    '''
    function evaluate_constant_height - evaluate the gaussian mixture at a given point(s). evaluate point at each of the gaussian
        distributions, then add them all up. IN THIS CASE, disregard constant in front of the exponent of the gaussian.
        Only evaluate the exponent part of the formula, then multiply by magnitude and add.
    @param x - tensor of (n,3) points to evaluate the gaussians
    @return - toret tensor of (n,1) with the combined source values.
    '''
    def evaluate_constant_height(self,x):
        n = len(x)
        assert x.shape[1] == self.spatial_dim
        base = torch.zeros(n,1)
        # print(self.magnitude)
        tensor = torch.zeros(n,self.num_gaussian).float()
        '''for each source:'''
        for i in range(self.num_gaussian):
            '''take location of that source'''
            source_pts = self.mean[i].view(1,-1).repeat((len(x),1))
            source_stdev = self.st_dev[i].view(1,-1).repeat((len(x),1))
            assert source_pts.shape == x.shape
            # assert source_stdev.shape == (len(x)*self.num_gaussian,self.spatial_dim)
            assert source_stdev.shape == source_pts.shape
            assert source_stdev.shape == (n,self.spatial_dim)
            # res = self.magnitude[i]*1/(((2*torch.pi)**(self.spatial_dim/2))*torch.prod(source_stdev,dim=1))*torch.exp(-torch.sum(torch.square(x - source_pts)/(2*source_stdev**2),dim=1))
            res = torch.exp(torch.sum(-(x - source_pts)**2/(2*source_stdev**2),dim=1))
            tensor[:,i] = res
        '''multiply gaussians by magnitudes. Need matrix multiplication for autograd to pick up.'''
        toret = (torch.clamp(self.magnitude,0,10) @ torch.transpose(tensor.float(),0,1)).view(-1,1)
        # print(torch.max(toret))
        return toret
    
    '''
    function source_points - generate sample points around each non-zero source.
    @param n - number of points per source
    @param t_max - maximum value of t to include in samples.
    '''
    def source_points(self,n,t_max):
        source_inputs_ls = torch.empty(0,4)
        # for each source
        for i in range(len(self.magnitude)):
            if self.magnitude[i] > .000001:
                #was using numpy's tile, but apparently torch broadcasting does it automatically.
                rand_source = self.mean[i] + torch.randn(n, 3, device=self.mean.device) * self.st_dev[i]
                #generate random time values
                rand_time = torch.rand(n,1)*t_max # Shape: (61,)
                #combine time and random source points
                source_inputs = torch.cat([rand_time,rand_source],dim=1)
                #add points of this source to what we are returning. 
                source_inputs_ls= torch.cat([source_inputs_ls, source_inputs])

        return source_inputs_ls.float()


# gm = Gaussian_Mixture([[0,0,0],[1,1,1],[2,2,2]],[[1,1,1],[1,1,1],[1,1,1]])
# tensor = torch.rand(3,3)
# bt = time.time()
# print(gm.compute(tensor))
# et = time.time()
# print(et - bt)
# mean = [[0,0,0],[0,0,0],[100,100,100],[100,100,100],[100,100,100],[100,100,100],[100,100,100]]
# var = [[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]]
# magnitude = [1,1,1,1,1,1,1]



# mean = np.array([[0,0,0],[0,0,0]])
# var = [[.025,.025,.025],[.025,.025,.025]]
# magnitude = [1,1]
# gm = Gaussian_Mixture(mean,var,magnitude,trainable=False)
# tensor = torch.rand(2,3)*.025
# tensor = torch.rand(2,3)*.025
# tensor = torch.tensor([[.01,.01,.01]])
# bt = time.time()
# x=gm.evaluate(tensor)
# et = time.time()
# print(et - bt)
# print(x)


# sgm = GaussianMixture(len(mean),covariance_type='full')
# non_zero_idx = np.array(magnitude) > 0
# sgm.weights_=np.array(magnitude)[non_zero_idx]
# sgm.means_ = np.array(mean)[non_zero_idx]
# sgm.covariances_ = np.array([np.eye(3)*.025 for _ in range(len(mean))])[non_zero_idx]
# sgm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(sgm.covariances_))
# source_term = torch.tensor(np.exp(sgm.score_samples(tensor.detach().cpu().numpy()))).view(tensor.shape[0],1)
# print(source_term)
# print(torch.mean(torch.abs(source_term - x)))
# # print(torch.tensor([1,2,3,4])*torch.tensor([1,2,3,4]))

# # print(torch.prod(torch.tensor([[1,2,3,4],[2,3,4,5]]),dim=1))

# # res = torch.arange(0,12).view(4,3)
# # print(res)
# # n = 2
# # idx = torch.arange(0,len(res))
# # filter = n*(idx % n) + idx//n
# # print(filter)
# # print(res[filter].view(-1,3,n))
# # print(res[filter].view(-1,3,n).sum(dim=1).sum(dim=1))

# mean = [[0,0,0]]
# stdev = [[.025,.025,.025]]
# magnitude = [1]

# mod = Gaussian_Mixture(mean,stdev,magnitude,False)
# print(mod.evaluate(torch.tensor([[.01,.01,.01]])))