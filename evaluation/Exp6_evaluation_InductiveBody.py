import os 
loc = os.getcwd()

import all_systems
import time 
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torch.nn.utils.prune as prune
import numpy as np
import os
import time
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


error = lambda x,y: np.sum(np.sqrt(np.sum((x-y)**2,0))/np.sqrt(np.sum(x**2,0))) # where x is the true vector and y is the approximated vector
errorzero = lambda x,y: np.sum(np.sqrt(np.sum((x-y)**2,0)))



# define model
def softplus(x):
    return torch.log(torch.exp(x)+1)

class newLinear(nn.Module):

    def __init__(self, input_size, output_size):
        # output_size is the size of one of the two parallel networks
        super(newLinear, self).__init__()
        self.input_size, self.output_size = input_size, output_size
        weight = torch.Tensor(self.input_size,self.output_size)
        self.weight = nn.Parameter(weight)
        bias = torch.Tensor(1,self.output_size)
        self.bias = nn.Parameter(bias)

        # initialise weights and bias
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) 
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init
        
    def forward(self, x):
        # print(self.weights, self.bias)
        # print("mul", torch.einsum('ijk,ikl->ijl', x, self.weights))
        # print("add", torch.add(torch.einsum('ijk,ikl->ijl', x, self.weights), self.bias))
        return torch.add(torch.einsum('jk,kl->jl', x, self.weight), self.bias)
        # return F.linear(x, self.weights, self.bias)


class HsepNet(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(HsepNet, self).__init__()
        self.hidden_layer_1a = newLinear( input_size, hidden_size1)  
        self.hidden_layer_1b = newLinear( hidden_size1, hidden_size2)  
        self.hidden_layer_1c = newLinear( hidden_size2, output_size)  
        self.hidden_layer_2a = newLinear( input_size, hidden_size1)  
        self.hidden_layer_2b = newLinear( hidden_size1, hidden_size2)  
        self.hidden_layer_2c = newLinear( hidden_size2, output_size)  
        
    def forward(self, x):
        # print("input", x.shape)
        # print(x)
        xa, xb = x[:,:int(x.shape[-1]/2)],x[:,int(x.shape[-1]/2):]
        # print("input", torch.sum(xa), torch.sum(xb))
        xa = softplus(self.hidden_layer_1a(xa))
        xb = softplus(self.hidden_layer_2a(xb)) 
        xa = softplus(self.hidden_layer_1b(xa)) 
        xb = softplus(self.hidden_layer_2b(xb)) 
        xa = self.hidden_layer_1c(xa)
        xb = self.hidden_layer_2c(xb)
        x = torch.sum(torch.stack((xa,xb)), dim = 0)
        # print("outp", torch.sum(x))
        return x

class sepNet(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(sepNet , self).__init__()
        self.hidden_layer_1 = splitBalancedLinear(input_size, hidden_size1)
        self.hidden_layer_2 = splitBalancedLinear(hidden_size1, hidden_size2)
        self.output_layer = splitBalancedLinear(hidden_size2, output_size)
        
    def forward(self, x):
        # print("input", x.shape)
        # print(x)
        x = torch.stack((x[:,:int(x.shape[-1]/2)],x[:,int(x.shape[-1]/2):]))
        # print(x)
        # print("initial", x.shape)
        x = softplus(self.hidden_layer_1(x)) 
        # print(x)
        # print("hl1", x.shape)
        x = softplus(self.hidden_layer_2(x)) 
        # print(x)
        # print("hl2", x.shape)
        x = self.output_layer(x)
        # print(x)
        # print("output", x.shape)
        x = torch.sum(x, dim = 0)
        return x


class splitBalancedLinear(nn.Module):

    def __init__(self, input_size, output_size):
        # output_size is the size of one of the two parallel networks
        super(splitBalancedLinear , self).__init__()
        self.input_size, self.output_size = input_size, output_size
        weights = torch.Tensor(2,self.input_size,self.output_size)
        self.weights = nn.Parameter(weights)
        bias = torch.Tensor(2,1,self.output_size)
        self.bias = nn.Parameter(bias)

        # initialise weights and bias
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) 
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init
        
    def forward(self, x):
        # print(self.weights, self.bias)
        # print("mul", torch.einsum('ijk,ikl->ijl', x, self.weights))
        # print("add", torch.add(torch.einsum('ijk,ikl->ijl', x, self.weights), self.bias))
        return torch.add(torch.einsum('ijk,ikl->ijl', x, self.weights), self.bias)
        # return F.linear(x, self.weights, self.bias)


if torch.cuda.is_available():
  device=torch.device('cuda:1')
else:
  device=torch.device('cpu')
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def get_grad(model, z,device):
  inputs=Variable(z.clone().detach()).requires_grad_(True).to(device)
  out=model(inputs.float())
  dH=torch.autograd.grad(out, inputs, grad_outputs=out.data.new(out.shape).fill_(1),create_graph=True)[0]
  return np.asarray([dH.detach().cpu().numpy()[:,i] for i in range(int(len(z[0])/2), int(len(z[0])))]), -np.asarray([dH.detach().cpu().numpy()[:,i] for i in range(int(len(z[0])/2))]) # negative dH/dq is dp/dt

n_sample = 10
pd.options.mode.chained_assignment = None  # default='warn'

for s in ["pendulum", "trigo", "arctan", "logarithm", "anisotropicoscillator2D", "henonheiles", "todalattice", "coupledoscillator"]:
    exec("sys = all_systems.%s" %s)
    dim = len(sys.spacedim)
    df = pd.DataFrame()
    z = torch.tensor(np.array(np.meshgrid(*[np.linspace(sys.spacedim[i][0], sys.spacedim[i][1], 10) for i in range(dim)],
                                      ))).reshape(len(sys.spacedim), 10**dim).transpose(1,0)
    df = pd.DataFrame(columns = ["seed","ini","model","loss","epochs","time"])

    for f in glob.glob(loc+"/Experiments/ResultsInductiveBody/%s/*.pt" %s):
        details = f.split("/")[-1].split("_")
        fvec = lambda z: np.concatenate([np.expand_dims(sys.f1(z),0), np.expand_dims(sys.f2(z),0)]) if len(sys.spacedim)==2 else np.concatenate([sys.f1(z), sys.f2(z)])
        if (len(details) == 5) & (details[0] == "SepNet"):
          if int(details[3]) > 0:
            net = sepNet(sys.sepshape1,sys.sepshape2,sys.sepshape3,1) 
            net.load_state_dict(torch.load(f))
            model = "sHNN-I"
            net = net.to(device)
            data = pd.Series({'seed':int(details[1]), 
                'ini':int(details[2]), 'model':model, 'epochs':int(details[3]),
                'loss':float(error(fvec(z), np.concatenate(get_grad(net, z, device)))/(n_sample**dim)), 
                "time":float(details[-1][:-3]), "tpe":float(details[-1][:-3])/int(details[3])})        
            df = pd.concat([df,data.to_frame().T], ignore_index = True)

    for f in glob.glob(loc+"/Experiments/ResultsInductiveBody/%s/*.pt" %s):
      details = f.split("/")[-1].split("_")
      fvec = lambda z: np.concatenate([np.expand_dims(sys.f1(z),0), np.expand_dims(sys.f2(z),0)]) if len(sys.spacedim)==2 else np.concatenate([sys.f1(z), sys.f2(z)])
      if (len(details) == 5) & (details[0] == "HetNet"):
        if int(details[3]) > 0:
          net = HsepNet(sys.sepshape1,sys.sepshape2,sys.sepshape3,1) 
          net.load_state_dict(torch.load(f))
          model = "sHNN-I (H)"
          net = net.to(device)
          eval = net(Variable(z.clone()).to(device).float())-net(torch.ones(z.shape).to(device))
          data = pd.Series({'seed':int(details[1]), 
                'ini':int(details[2]), 'model':model, 'epochs':int(details[3]),
                'loss':float(error(fvec(z), np.concatenate(get_grad(net, z, device)))/(n_sample**dim)), 
                "time":float(details[-1][:-3]), "tpe":float(details[-1][:-3])/int(details[3])})        
          if  (int(details[3])>0):
             df = pd.concat([df,data.to_frame().T], ignore_index = True)

    
    df = df.groupby(by = ["ini","model"], as_index = False).agg(['mean','sem'])
    df = df.reset_index()
    df = df[df["ini"] == 512]
    print(df)
    print("%.2E (%.2E) & %.2E (%.2E)" %(df[df["model"] == "sHNN-I"]["loss"]["mean"].values[0], df[df["model"] == "sHNN-I"]["loss"]["sem"].values[0], 
					df[df["model"] == "sHNN-I (H)"]["loss"]["mean"].values[0], df[df["model"] == "sHNN-I (H)"]["loss"]["sem"].values[0], 	))	
    print("%.2f (%.0f) & %.2f (%.0f)" %(df[df["model"] == "sHNN-I"]["time"]["mean"].values[0], df[df["model"] == "sHNN-I"]["epochs"]["mean"].values[0], 
					df[df["model"] == "sHNN-I (H)"]["time"]["mean"].values[0], df[df["model"] == "sHNN-I (H)"]["epochs"]["mean"].values[0], 	))		




