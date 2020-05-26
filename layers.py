import torch
import numpy as np
import time

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.nn import Parameter
from torchvision.models import vgg16_bn, vgg16
import torchvision



class SparseConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(SparseConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.log_sigma2 = Parameter(-10 * torch.ones_like(self.weight))
        self.reset_parameter()
        self.thresh = 3
    def reset_parameter(self):
        self.bias.data.zero_()
        self.weight.data.normal_(0, 0.02)  
        self.log_sigma2.data.fill_(-10)       
    def forward(self, input):
        if self.training:
            mu = F.conv2d(input, self.weight, self.bias, self.stride,
                self.padding, self.dilation, self.groups)
            std = torch.sqrt(F.conv2d(input * input, torch.exp(self.log_sigma2), None, self.stride,\
                self.padding, self.dilation, self.groups) + 1e-16)
            eps = std.data.new(std.size()[:2] + (1,1)).normal_()
            return mu + std * eps
        return F.conv2d(input, (self.log_alpha < self.thresh).float() * self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
    
    def kl_reg(self):
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        kl = k1 * torch.sigmoid(k2 + k3 * self.log_alpha) - 0.5 * torch.log1p(torch.exp(-1 * self.log_alpha)) - k1
        a = -1* torch.sum(kl)
        return a
    @property    
    def zero_num(self):
        return (self.log_alpha > self.thresh).sum().float()
    @property    
    def sparsity(self):
        return (self.zero_num / self.num_el).item()
    @property
    def num_el(self):
        return self.log_alpha.numel()

    @property    
    def log_alpha(self):
        eps = 1e-8
        return  torch.clamp(self.log_sigma2 - 2.0 * torch.log(torch.abs(self.weight) + eps), -10, 10)
class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = Parameter(torch.Tensor(out_features, in_features))
        self.log_sigma2 = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(1, out_features))
        self.thresh = 3
        self.reset_parameters()
    def reset_parameters(self):
        self.bias.data.zero_()
        self.W.data.normal_(0, 0.02)
        self.log_sigma2.data.fill_(-10)             
    def forward(self, x):
        if self.training:
            lrt_mean =  F.linear(x, self.W) + self.bias
            lrt_std = torch.sqrt(F.linear(x * x, torch.exp(self.log_sigma2)) + 1e-8)
            eps = lrt_std.data.new(lrt_std.size()).normal_()
            return lrt_mean + lrt_std * eps

        return F.linear(x, self.W * (self.log_alpha < self.thresh).float()) + self.bias
    @property    
    def zero_num(self):
        return (self.log_alpha > self.thresh).sum().float()
    @property    
    def sparsity(self):
        return (self.zero_num / self.num_el).item()
    @property
    def num_el(self):
        return self.log_alpha.numel()
    def kl_reg(self):
        # Return KL here -- a scalar 
        d = torch.device('cuda' if self.log_sigma2.is_cuda else 'cpu')
        k1, k2, k3 = torch.Tensor([0.63576]).to(d), torch.Tensor([1.8732]).to(d), torch.Tensor([1.48695]).to(d)
        kl = k1 * torch.sigmoid(k2 + k3 * self.log_alpha) - 0.5 * torch.log1p(torch.exp(-1 * self.log_alpha)) - k1
        a = -1* torch.sum(kl)
        return a
    @property    
    def log_alpha(self):
        eps = 1e-10
        return  torch.clamp(self.log_sigma2 - 2.0 * torch.log(torch.abs(self.W)+eps), -10, 10)
    
    def phi(x):
    if torch.cuda.is_available():
        loc = torch.cuda.FloatTensor([0.0])
        scale=torch.cuda.FloatTensor([1.0])
    else:
        loc = torch.FloatTensor([0.0])
        scale=torch.FloatTensor([1.0])
    normal = torch.distributions.normal.Normal(loc=loc, scale=scale)
    return normal.cdf(x)
def phi_inv(x):
    if torch.cuda.is_available():
        loc = torch.cuda.FloatTensor([0.0])
        scale=torch.cuda.FloatTensor([1.0])
    else:
        loc = torch.FloatTensor([0.0])
        scale=torch.FloatTensor([1.0])
    normal = torch.distributions.normal.Normal(loc=loc, scale=scale)
    return normal.icdf(x)
def mean_truncated_log_normal_reduced(mu, sigma, a, b):
    alpha = (a - mu)/sigma
    beta = (b - mu)/sigma
    z = phi(beta) - phi(alpha)
    mean = erfcx((sigma-beta)/(2 ** 0.5))*torch.exp(b-beta*beta/2)
    mean = mean - erfcx((sigma-alpha)/(2 ** 0.5))*torch.exp(a-alpha*alpha/2)
    mean = mean/(2*z)
    return mean
def snr_truncated_log_normal(mu, sigma, a, b):
    alpha = (a - mu)/sigma
    beta = (b - mu)/sigma
    z = phi(beta) - phi(alpha)
    ratio = erfcx((sigma-beta)/(2 ** 0.5))*torch.exp((b-mu)-beta**2/2.0)
    ratio = ratio - erfcx((sigma-alpha)/2 ** 0.5)*torch.exp((a-mu)-alpha**2/2.0)
    denominator = 2*z*erfcx((2.0*sigma-beta)/2 ** 0.5)*torch.exp(2.0*(b-mu)-beta**2/2.0)
    denominator = denominator - 2*z*erfcx((2.0*sigma-alpha)/(2 ** 0.5))*torch.exp(2.0*(a-mu)-alpha**2/2.0)
    denominator = denominator - ratio**2
    ratio = ratio/torch.sqrt(1e-8 + denominator)
    return ratio
def erfcx(x):
    """M. M. Shepherd and J. G. Laframboise,
       MATHEMATICS OF COMPUTATION 36, 249 (1981)
    """
    x = x.cpu()
    K = 3.75
    y = (torch.abs(x)-K) / (torch.abs(x)+K)
    y2 = 2.0*y
    (d, dd) = (-0.4e-20, 0.0)
    (d, dd) = (y2 * d - dd + 0.3e-20, d)
    (d, dd) = (y2 * d - dd + 0.97e-19, d)
    (d, dd) = (y2 * d - dd + 0.27e-19, d)
    (d, dd) = (y2 * d - dd + -0.2187e-17, d)
    (d, dd) = (y2 * d - dd + -0.2237e-17, d)
    (d, dd) = (y2 * d - dd + 0.50681e-16, d)
    (d, dd) = (y2 * d - dd + 0.74182e-16, d)
    (d, dd) = (y2 * d - dd + -0.1250795e-14, d)
    (d, dd) = (y2 * d - dd + -0.1864563e-14, d)
    (d, dd) = (y2 * d - dd + 0.33478119e-13, d)
    (d, dd) = (y2 * d - dd + 0.32525481e-13, d)
    (d, dd) = (y2 * d - dd + -0.965469675e-12, d)
    (d, dd) = (y2 * d - dd + 0.194558685e-12, d)
    (d, dd) = (y2 * d - dd + 0.28687950109e-10, d)
    (d, dd) = (y2 * d - dd + -0.63180883409e-10, d)
    (d, dd) = (y2 * d - dd + -0.775440020883e-09, d)
    (d, dd) = (y2 * d - dd + 0.4521959811218e-08, d)
    (d, dd) = (y2 * d - dd + 0.10764999465671e-07, d)
    (d, dd) = (y2 * d - dd + -0.218864010492344e-06, d)
    (d, dd) = (y2 * d - dd + 0.774038306619849e-06, d)
    (d, dd) = (y2 * d - dd + 0.4139027986073010e-05, d)
    (d, dd) = (y2 * d - dd + -0.69169733025012064e-04, d)
    (d, dd) = (y2 * d - dd + 0.490775836525808632e-03, d)
    (d, dd) = (y2 * d - dd + -0.2413163540417608191e-02, d)
    (d, dd) = (y2 * d - dd + 0.9074997670705265094e-02, d)
    (d, dd) = (y2 * d - dd + -0.26658668435305752277e-01, d)
    (d, dd) = (y2 * d - dd + 0.59209939998191890498e-01, d)
    (d, dd) = (y2 * d - dd + -0.84249133366517915584e-01, d)
    (d, dd) = (y2 * d - dd + -0.4590054580646477331e-02, d)
    d = y * d - dd + 0.1177578934567401754080e+01
    result = d/(1.0+2.0*torch.abs(x))
    result[result!=result] = 1.0
    result[result == float("Inf")] = 1.0

    negative_mask = torch.zeros(x.size())
    negative_mask[x<=0] = 1.0
    positive_mask = torch.zeros(x.size())
    positive_mask[x>0] = 1.0
    negative_result = 2.0*torch.exp(x*x)-result
    negative_result[negative_result!=negative_result] = 1.0
    negative_result[negative_result == float("Inf")] = 1.0
    result = negative_mask * negative_result + positive_mask * result
    result = result.cuda()
    return result
class StructDropout_lognorm(nn.Module):
    def __init__(self, channels):
        super(StructDropout_lognorm, self).__init__()
        self.channels = channels
        self.a = -20
        self.b = 0
        self.mu = Parameter(torch.zeros(self.channels))
        self.log_sigma = Parameter(-5 * torch.ones(self.channels))
        self.thresh = 1.
    def train(self, mode=True):
        self.training = mode
        if mode==False:
            log_sigma = torch.clamp(self.log_sigma,min=-20,max=5)
            mu = torch.clamp(self.mu,min=-20,max=5)
            sigma = torch.exp(log_sigma)
            self.mean_teta = mean_truncated_log_normal_reduced(mu.detach(), sigma.detach(), self.a, self.b)
            self.snr = snr_truncated_log_normal(mu.detach(), sigma.detach(), self.a, self.b)
        return self
    def sample_truncated_normal(self, mu, sigma, a, b):
        alpha = (a - mu)/sigma
        beta = (b - mu)/sigma
        uniform = torch.distributions.uniform.Uniform(low=0.0,high=1.0)
        sampled_uniform = uniform.sample(mu.size())
        sampled_uniform = sampled_uniform.cuda()
        gamma = phi(alpha)+sampled_uniform*(phi(beta)-phi(alpha))
        return torch.clamp(phi_inv(torch.clamp(gamma, min=1e-5, max=1.0-1e-5))*sigma+mu, min=a, max=b)
    @property    
    def zero_num(self):
        mu = self.mu
        sigma = torch.exp(self.log_sigma)
        self.snr = snr_truncated_log_normal(mu.detach(), sigma.detach(), self.a, self.b)
        return (self.snr <= self.thresh).sum().float()
    @property    
    def sparsity(self):
        return (self.zero_num / self.num_el).item(), self.snr.min().item(), self.snr.max().item()
    @property
    def num_el(self):
        return self.mu.numel()
    def forward(self, x):
        if self.training:
            log_sigma = torch.clamp(self.log_sigma,min=-20,max=5)
            mu = torch.clamp(self.mu,min=-20,max=5)
            sigma = torch.exp(log_sigma)
            alpha = (self.a-mu)/sigma
            beta = (self.b-mu)/sigma
            teta = torch.exp(self.sample_truncated_normal(mu, sigma, self.a, self.b))
            if (x.size().__len__() == 4):
                teta = teta.unsqueeze(1)
                teta = teta.unsqueeze(2)
            return x * teta
        mask = (self.snr > self.thresh)
        teta = self.mean_teta * mask
        if (x.size().__len__() == 4):
            teta = teta.unsqueeze(1)
            teta = teta.unsqueeze(2)
        return  teta * x
    def pdf(self, x):
        if torch.cuda.is_available():
            loc = torch.cuda.FloatTensor([0.0])
            scale=torch.cuda.FloatTensor([1.0])
        else:
            loc = torch.FloatTensor([0.0])
            scale=torch.FloatTensor([1.0])
        normal = torch.distributions.normal.Normal(loc=loc,scale=scale)
        return torch.exp(normal.log_prob(x))
    def kl_reg(self):
        log_sigma = torch.clamp(self.log_sigma,min=-20,max=5)
        mu = torch.clamp(self.mu,min=-20,max=5)
        sigma = torch.exp(log_sigma)
        alpha = (self.a-mu)/sigma
        beta = (self.b-mu)/sigma
        z = phi(beta) - phi(alpha)
        kl = -log_sigma - torch.log(z) - (alpha * self.pdf(alpha) - beta * self.pdf(beta)) / (2.0 * z)
        kl = kl + np.log(self.b - self.a) - np.log(2.0 * np.pi * np.e) / 2.0
        return kl.sum()