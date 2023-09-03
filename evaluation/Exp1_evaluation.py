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



# define model
def softplus(x):
    return torch.log(torch.exp(x)+1)


# PINN
class Net(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Net , self).__init__()
        self.hidden_layer_1 = nn.Linear( input_size, hidden_size, bias=True)
        self.hidden_layer_2 = nn.Linear( hidden_size, hidden_size, bias=True)
        self.output_layer = nn.Linear( hidden_size, output_size , bias=True)
        
    def forward(self, x):
        x = softplus(self.hidden_layer_1(x)) # F.relu(self.hidden_layer_1(x)) # 
        x = softplus(self.hidden_layer_2(x)) # F.relu(self.hidden_layer_2(x)) # 
        x = self.output_layer(x)

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





if torch.cuda.is_available():
  device=torch.device('cuda:0')
else:
  device=torch.device('cpu')
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

df = pd.DataFrame(columns = ["seed","ini","model","loss","epoch","time"])

def get_grad(model, z,device):
  inputs=Variable(z.clone().detach()).requires_grad_(True).to(device)
  out=model(inputs.float())
  dH=torch.autograd.grad(out, inputs, grad_outputs=out.data.new(out.shape).fill_(1),create_graph=True)[0]
  return np.asarray([dH.detach().cpu().numpy()[:,i] for i in range(int(len(z[0])/2), int(len(z[0])))]), -np.asarray([dH.detach().cpu().numpy()[:,i] for i in range(int(len(z[0])/2))]) # negative dH/dq is dp/dt

n_sample = 10

for s in ["pendulum","trigo","arctan","logarithm","anisotropicoscillator2D","henonheiles","todalattice","coupledoscillator"]:
    exec("sys = all_systems.%s" %s)
    dim = len(sys.spacedim)
    df = pd.DataFrame(columns = ["seed","ini","model","loss","epochs","time"])
    z = torch.tensor(np.array(np.meshgrid(*[np.linspace(sys.spacedim[i][0], sys.spacedim[i][1], 10) for i in range(dim)],
                                      ))).reshape(len(sys.spacedim), 10**dim).transpose(1,0)
    for f in glob.glob(loc+"/Experiments/Results/%s/*.pt" %s):
        details = f.split("/")[-1].split("_")
        fvec = lambda z: np.concatenate([np.expand_dims(sys.f1(z),0), np.expand_dims(sys.f2(z),0)]) if len(sys.spacedim)==2 else np.concatenate([sys.f1(z), sys.f2(z)])
        if (len(details) == 5) & (details[0] == "PINN"):
            net = Net(sys.netshape1,sys.netshape2,1)
            net.load_state_dict(torch.load(f))
            model = "HNN"
            net = net.to(device)
            data = pd.Series({'seed':int(details[1]), 
                'ini':int(details[2]), 'model':model, 'epochs':int(details[3]),
                'loss':float(error(fvec(z), np.concatenate(get_grad(net, z, device)))/(n_sample**dim)), 
                'Hloss':torch.mean((sys.H(z.clone().detach()) - net(Variable(z.clone()).to(device).float()).detach().cpu()[:,0])**2),
                "time":float(details[-1][:-3])})         
            if  (int(details[3])>0):
              df = pd.concat([df,data.to_frame().T], ignore_index = True)
        elif (len(details) == 5) & (details[0] == "sepNN"):
            net = sepNet(sys.sepshape1,sys.sepshape2,sys.sepshape3,1) 
            net.load_state_dict(torch.load(f))
            model = "sHNN-I"
            net = net.to(device)
            data = pd.Series({'seed':int(details[1]), 
                'ini':int(details[2]), 'model':model, 'epochs':int(details[3]),
                'loss':float(error(fvec(z), np.concatenate(get_grad(net, z, device)))/(n_sample**dim)), 
                'Hloss':torch.mean((sys.H(z.clone().detach()) - net(Variable(z.clone()).to(device).float()).detach().cpu()[:,0])**2),
                "time":float(details[-1][:-3])})           
            if  (int(details[3])>0):
              df = pd.concat([df,data.to_frame().T], ignore_index = True)


    for f in glob.glob(loc+"/Experiments/ResultsLearningCoefficient/%s/*.pt" %s):
        details = f.split("/")[-1].split("_")
        fvec = lambda z: np.concatenate([np.expand_dims(sys.f1(z),0), np.expand_dims(sys.f2(z),0)]) if len(sys.spacedim)==2 else np.concatenate([sys.f1(z), sys.f2(z)])
        if (len(details) == 6) & (str(details[0]) == "PINN") & (str(details[1]) == "softsep-1.00"): # its a softsep PINN
            net = Net(sys.netshape1,sys.netshape2,1)
            net.load_state_dict(torch.load(f))
            model = "sHNN-L"
            net = net.to(device)
            data = pd.Series({'seed':int(details[2]), 
                'ini':int(details[3]), 'model':model, 'epochs':int(details[4]),
                'loss':float(error(fvec(z), np.concatenate(get_grad(net, z, device)))/(n_sample**dim)), 
                'Hloss':torch.mean((sys.H(z.clone().detach()) - net(Variable(z.clone()).to(device).float()).detach().cpu()[:,0])**2),
                "time":float(details[-1][:-3])})
            if  (int(details[4])>0):
              df = pd.concat([df,data.to_frame().T], ignore_index = True)


    for f in glob.glob(loc+"/Experiments/ResultsObservational/%s/*.pt" %s):
        details = f.split("/")[-1].split("_")
        fvec = lambda z: np.concatenate([np.expand_dims(sys.f1(z),0), np.expand_dims(sys.f2(z),0)]) if len(sys.spacedim)==2 else np.concatenate([sys.f1(z), sys.f2(z)])
        if (len(details) == 5) & (details[0] == "obsNet2"):
            net = Net(sys.netshape1,sys.netshape2,1)
            net.load_state_dict(torch.load(f))
            model = "sHNN-O"
            net = net.to(device)
            data = pd.Series({'seed':int(details[1]), 
                'ini':int(details[2]), 'model':model, 'epochs':int(details[3]),
                'loss':float(error(fvec(z), np.concatenate(get_grad(net, z, device)))/(n_sample**dim)), 
                'Hloss':torch.mean((sys.H(z.clone().detach()) - net(Variable(z.clone()).to(device).float()).detach().cpu()[:,0])**2),
                "time":float(details[-1][:-3])})         
            if  (int(details[3])>0):
              df = pd.concat([df,data.to_frame().T], ignore_index = True)


    df = df.groupby(by = ["ini","model"], as_index = False).agg(['mean','sem'])
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df = df.reset_index()
    df = df[df["ini"]==512]
    df = df[["model","loss_mean","loss_sem","Hloss_mean","Hloss_sem","time_mean","epochs_mean"]]
    print(df)

    print("%s & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E)" %(s, df[df["model"]=="HNN"]["loss_mean"], df[df["model"]=="HNN"]["loss_sem"], 
								df[df["model"]=="sHNN-O"]["loss_mean"], df[df["model"]=="sHNN-O"]["loss_sem"],
								df[df["model"]=="sHNN-L"]["loss_mean"], df[df["model"]=="sHNN-L"]["loss_sem"],
								df[df["model"]=="sHNN-I"]["loss_mean"], df[df["model"]=="sHNN-I"]["loss_sem"],))
    print("%s & 0.00 \\%% & %.2f \\%% & %.2f \\%% & %.2f \\%% " %(s, (1-df[df["model"]=="sHNN-O"]["loss_mean"].values/df[df["model"]=="HNN"]["loss_mean"].values)*100, 
								(1-df[df["model"]=="sHNN-L"]["loss_mean"].values/df[df["model"]=="HNN"]["loss_mean"].values)*100, 
								(1-df[df["model"]=="sHNN-I"]["loss_mean"].values/df[df["model"]=="HNN"]["loss_mean"].values)*100,))
    print("%s & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E)" %(s, df[df["model"]=="HNN"]["Hloss_mean"], df[df["model"]=="HNN"]["Hloss_sem"], 
								df[df["model"]=="sHNN-O"]["Hloss_mean"], df[df["model"]=="sHNN-O"]["Hloss_sem"],
								df[df["model"]=="sHNN-L"]["Hloss_mean"], df[df["model"]=="sHNN-L"]["Hloss_sem"],
								df[df["model"]=="sHNN-I"]["Hloss_mean"], df[df["model"]=="sHNN-I"]["Hloss_sem"],))
    print("%s & 0.00 \\%% & %.2f \\%% & %.2f \\%% & %.2f \\%% " %(s, (1-df[df["model"]=="sHNN-O"]["Hloss_mean"].values/df[df["model"]=="HNN"]["Hloss_mean"].values)*100, 
								(1-df[df["model"]=="sHNN-L"]["Hloss_mean"].values/df[df["model"]=="HNN"]["Hloss_mean"].values)*100, 
								(1-df[df["model"]=="sHNN-I"]["Hloss_mean"].values/df[df["model"]=="HNN"]["Hloss_mean"].values)*100,))
	
    print("%s & %.2f (%s) & %.2f (%s) & %.2f (%s) & %.2f (%s)" %(s, df[df["model"]=="HNN"]["time_mean"].values, df[df["model"]=="HNN"]["epochs_mean"].values[0], 
								df[df["model"]=="sHNN-O"]["time_mean"].values, df[df["model"]=="sHNN-O"]["epochs_mean"].values[0],
								df[df["model"]=="sHNN-L"]["time_mean"].values, df[df["model"]=="sHNN-L"]["epochs_mean"].values[0],
								df[df["model"]=="sHNN-I"]["time_mean"].values, df[df["model"]=="sHNN-I"]["epochs_mean"].values[0],))




