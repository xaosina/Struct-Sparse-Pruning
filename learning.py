import math
import torch
import numpy as np
import time

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.nn import Parameter
from torchvision import datasets, transforms
from torchvision.models import vgg16_bn, vgg16
import torchvision
from tqdm.notebook import tqdm as tqdm_notebook
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Struct_like(nn.Module):
    def __init__(self, model, lognorm=False):
      super(Struct_like, self).__init__()
      if lognorm == True:
          cl = StructDropout_lognorm
      else:
          cl = StructDropout_normal
      layers = []
      for layer in model.features:
          if isinstance(layer, torch.nn.modules.conv.Conv2d):
              layers += [layer, cl(layer.out_channels)]
          else:
              layers += [layer]
      self.features = nn.Sequential(*layers)
      layers = []
      for layer in model.classifier:
          if isinstance(layer, torch.nn.modules.linear.Linear):
              layers += [cl(layer.in_features), layer]
          else:
              layers += [layer]
      self.classifier = nn.Sequential(*layers)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(self.classifier(x), dim=1)
        return x

class Sparse_like(nn.Module):
    def __init__(self, model):
      super(Sparse_like, self).__init__()
      layers = []
      for layer in model.features:
          if isinstance(layer, torch.nn.modules.conv.Conv2d):
              ex = SparseConv2d(layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride,
                  layer.padding, layer.dilation, layer.groups, bias=True)
              ex.weight = layer.weight
              ex.bias = layer.bias
              layers += [ex]
          else:
              layers += [layer]
      self.features = nn.Sequential(*layers)
      layers = []
      for layer in model.classifier:
          if isinstance(layer, torch.nn.modules.linear.Linear):
              ex = SparseLinear(layer.in_features, layer.out_features)
              ex.W = layer.weight
              ex.bias = layer.bias
              layers += [ex]
          else:
              layers += [layer]
      self.classifier = nn.Sequential(*layers)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(self.classifier(x), dim=1)
        return x

class SGVLB(nn.Module):
    def __init__(self, net, train_size):
        super(SGVLB, self).__init__()
        self.train_size = train_size
        self.net = net
    def forward(self, input, target, kl_weight=0.0):
        assert not target.requires_grad
        a = F.cross_entropy(input, target)
        self.cr = a.item()
        #if kl_weight==0:
        #    return a
        kl = 0.0
        for module in self.net.features.children():
            if hasattr(module, 'kl_reg'):
                kl = kl + module.kl_reg()
        for module in self.net.classifier.children():
            if hasattr(module, 'kl_reg'):
                kl = kl + module.kl_reg()
        return (a * self.train_size + (kl_weight * kl))
    
class mywork():
    def __init__(self, model, train, test):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(self.device)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[100000000], gamma=1)
        
        self.trainset = train
        self.testset = test
        self.loss = SGVLB(model, len(self.trainset.dataset))
        self.model = model
        self.kl_weight = 0
        

    def train(self, epoch=300, kl = None, scaling = 0, lr = 1e-4, kl_lr=1, scheduler='lambda'):
        self.model = self.model.to(self.device)
        self.optim = optim.Adam(
                [{'params':layer, 'lr':kl_lr * lr} if (name.split('.')[-1]=='mu' or name.split('.')[-1]=='log_sigma') else \
                {'params':layer} for name, layer in self.model.named_parameters() if name.split('.')[-1] != 'bias'], lr=lr, weight_decay=1e-5, betas=(0.95,0.999))
        if scheduler != 'lambda':
            self.scheduler = ReduceLROnPlateau(self.optim,'min',factor=0.5 ,patience=1, verbose =True)
        else:
            lamb = lambda epoch: 1 if epoch < 250 else 1-(epoch-250)/50
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optim, lamb)
        if kl!=None:
            self.kl_weight = kl
        for e in tqdm_notebook(range(0,epoch)):
            self.model.train()
            self.kl_weight = min(self.kl_weight+ scaling, 1)
            running_loss = 0.0
            ce_loss = 0.0
            i = 0
            for data, target in tqdm_notebook(self.trainset):
                data, target = data.to(self.device), target.to(self.device)
                self.optim.zero_grad()

                output = self.model(data)
                pred = output.data.max(1)[1] 
                
                loss = self.loss(output, target, self.kl_weight)
                loss.backward()
                self.optim.step()
                running_loss += loss.item()
                ce_loss += self.loss.cr
            print('[%d] loss: %.8f' %
              (e + 1, running_loss / len(self.trainset.dataset)), '| entropy=', self.loss.train_size * ce_loss / len(self.trainset.dataset))
            self.test()
            if scheduler != 'lambda':
                self.scheduler.step(running_loss)
            else:
                self.scheduler.step()
    def test(self, thresh=None):
        if thresh!=None:
           remember = self.change_thresh(thresh)
        self.model.eval()
        test_loss, test_acc = 0, 0
        for data, target in tqdm_notebook(self.testset):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            pred = output.data.max(1)[1] 
            test_acc += np.sum(pred.cpu().numpy() == target.cpu().numpy())
        print('accuracy= ', 100 * test_acc/len(self.testset.dataset), '%')
        self.sparsity()
        if thresh!=None:
            self.change_thresh(remember)
        return test_acc/len(self.testset.dataset)
    def change_thresh(self, newt):
        if newt==None:
            pass
        for module in self.model.features.children():
            if hasattr(module, 'kl_reg'):
                k = module.thresh
                module.thresh = newt
        for module in self.model.classifier.children():
            if hasattr(module, 'kl_reg'):
                k = module.thresh
                module.thresh = newt
        return k
    def sparsity(self):
        sp = 0
        s = 0
        for layer in self.model.features:
                if hasattr(layer, 'kl_reg'):
                    sp += layer.zero_num
                    s += layer.num_el
                    print(layer,'--', layer.sparsity)
        for layer in self.model.classifier:
                if hasattr(layer, 'kl_reg'):
                    sp += layer.zero_num
                    s += layer.num_el
                    print(layer,'--', layer.sparsity)
        if s==0:
            return 
        print('Total sparsity = ', sp / s)