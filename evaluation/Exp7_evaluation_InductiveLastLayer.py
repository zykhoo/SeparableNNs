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

class sepNet1(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(sepNet1 , self).__init__()
        self.hidden_layer_1 = splitBalancedLinear(input_size, hidden_size1)
        self.hidden_layer_2 = splitBalancedLinear(hidden_size1, hidden_size2)
        self.output_layer = splitBalancedLinear(hidden_size2, output_size)
        self.output1_layer = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.output1_layer.weight = nn.Parameter(torch.ones(self.output1_layer.weight.shape))

        
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
        x = torch.squeeze(self.output_layer(x), dim = 2)
        self.output1_layer.requires_grad_(False)
        x = self.output1_layer(torch.transpose(x,0,1))
        # print(x)
        # print("output", x.shape)
        # x = torch.sum(x, dim = 0)
        return x


class sepNet2(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(sepNet2 , self).__init__()
        self.hidden_layer_1 = splitBalancedLinear(input_size, hidden_size1)
        self.hidden_layer_2 = splitBalancedLinear(hidden_size1, hidden_size2)
        self.output_layer = splitBalancedLinear(hidden_size2, output_size)
        self.output1_layer = nn.Linear(2, 1, bias=True)
        with torch.no_grad():
            self.output1_layer.weight = nn.Parameter(torch.ones(self.output1_layer.weight.shape))

        
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
        x = torch.squeeze(self.output_layer(x), dim = 2)
        self.output1_layer.weight.requires_grad_(False)
        self.output1_layer.bias.requires_grad_(True)
        x = self.output1_layer(torch.transpose(x,0,1))
        # print(x)
        # print("output", x.shape)
        # x = torch.sum(x, dim = 0)
        return x

class sepNet3(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(sepNet3, self).__init__()
        self.hidden_layer_1 = splitBalancedLinear(input_size, hidden_size1)
        self.hidden_layer_2 = splitBalancedLinear(hidden_size1, hidden_size2)
        self.output_layer = splitBalancedLinear(hidden_size2, output_size)
        self.output1_layer = nn.Linear(2, 1, bias=False)

        
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
        x = torch.squeeze(self.output_layer(x), dim = 2)
        x = self.output1_layer(torch.transpose(x,0,1))
        # print(x)
        # print("output", x.shape)
        # x = torch.sum(x, dim = 0)
        return x


class sepNet4(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(sepNet4, self).__init__()
        self.hidden_layer_1 = splitBalancedLinear(input_size, hidden_size1)
        self.hidden_layer_2 = splitBalancedLinear(hidden_size1, hidden_size2)
        self.output_layer = splitBalancedLinear(hidden_size2, output_size)
        self.output1_layer = nn.Linear(2, 1, bias=True)

        
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
        x = torch.squeeze(self.output_layer(x), dim = 2)
        x = self.output1_layer(torch.transpose(x,0,1))
        # print(x)
        # print("output", x.shape)
        # x = torch.sum(x, dim = 0)
        return x


class sepNet5(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(sepNet5, self).__init__()
        self.hidden_layer_1 = splitBalancedLinear(input_size, hidden_size1)
        self.hidden_layer_2 = splitBalancedLinear(hidden_size1, hidden_size2)
        self.output_layer = splitBalancedLinear(hidden_size2, output_size)
        self.output1_layer = nn.Linear(2, 1, bias=True)

        
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
        x = torch.squeeze(self.output_layer(x), dim = 2)
        x = softplus(self.output1_layer(torch.transpose(x,0,1)))
        return x


if torch.cuda.is_available():
  device=torch.device('cuda:1')
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

for s in ["pendulum", "trigo", "arctan", "logarithm", "anisotropicoscillator2D", "henonheiles", "todalattice", "coupledoscillator"]:
    exec("sys = all_systems.%s" %s)
    dim = len(sys.spacedim)
    df = pd.DataFrame(columns = ["seed","ini","model","loss","epochs","time"])
    z = torch.tensor(np.array(np.meshgrid(*[np.linspace(sys.spacedim[i][0], sys.spacedim[i][1], 10) for i in range(dim)],
                                      ))).reshape(len(sys.spacedim), 10**dim).transpose(1,0)
    fvec = lambda z: np.concatenate([np.expand_dims(sys.f1(z),0), np.expand_dims(sys.f2(z),0)]) if len(sys.spacedim)==2 else np.concatenate([sys.f1(z), sys.f2(z)])
    df = pd.DataFrame(columns = ["seed","ini","model","loss","epochs","time"])
    for f in glob.glob(loc+"/Experiments/ResultsInductiveLastLayer/%s/*.pt" %s):
        details = f.split("/")[-1].split("_")
        if (len(details) == 5) & (details[0] == "SepNet1") & (int(details[3])>0):
            net = sepNet1(sys.sepshape1,sys.sepshape2,sys.sepshape3,1) 
            net.load_state_dict(torch.load(f))
            model = "sHNN-I1"
            net = net.to(device)
            data = pd.Series({'seed':int(details[1]), 
                'ini':int(details[2]), 'model':model, 'epochs':int(details[3]),
                'loss':float(error(fvec(z), np.concatenate(get_grad(net, z, device)))/(n_sample**dim)), 
                'Hloss':torch.mean((sys.H(z.clone().detach()) - net(Variable(z.clone()).to(device).float()).detach().cpu()[:,0])**2),
                "time":float(details[-1][:-3])})           
            df = pd.concat([df,data.to_frame().T], ignore_index = True)
        elif (len(details) == 5) & (details[0] == "SepNet0") & (int(details[3])>0):
            net = sepNet(sys.sepshape1,sys.sepshape2,sys.sepshape3,1) 
            net.load_state_dict(torch.load(f))
            model = "sHNN-I0"
            net = net.to(device)
            data = pd.Series({'seed':int(details[1]), 
                'ini':int(details[2]), 'model':model, 'epochs':int(details[3]),
                'loss':float(error(fvec(z), np.concatenate(get_grad(net, z, device)))/(n_sample**dim)), 
                'Hloss':torch.mean((sys.H(z.clone().detach()) - net(Variable(z.clone()).to(device).float()).detach().cpu()[:,0])**2),
                "time":float(details[-1][:-3])})           
            df = pd.concat([df,data.to_frame().T], ignore_index = True)
        elif (len(details) == 5) & (details[0] == "SepNet2") & (int(details[3])>0):
            net = sepNet2(sys.sepshape1,sys.sepshape2,sys.sepshape3,1) 
            net.load_state_dict(torch.load(f))
            model = "sHNN-I2"
            net = net.to(device)
            data = pd.Series({'seed':int(details[1]), 
                'ini':int(details[2]), 'model':model, 'epochs':int(details[3]),
                'loss':float(error(fvec(z), np.concatenate(get_grad(net, z, device)))/(n_sample**dim)), 
                'Hloss':torch.mean((sys.H(z.clone().detach()) - net(Variable(z.clone()).to(device).float()).detach().cpu()[:,0])**2),
                "time":float(details[-1][:-3])})            
            df = pd.concat([df,data.to_frame().T], ignore_index = True)
        elif (len(details) == 5) & (details[0] == "SepNet3") & (int(details[3])>0):
            net = sepNet3(sys.sepshape1,sys.sepshape2,sys.sepshape3,1) 
            net.load_state_dict(torch.load(f))
            model = "sHNN-I3"
            net = net.to(device)
            data = pd.Series({'seed':int(details[1]), 
                'ini':int(details[2]), 'model':model, 'epochs':int(details[3]),
                'loss':float(error(fvec(z), np.concatenate(get_grad(net, z, device)))/(n_sample**dim)), 
                'Hloss':torch.mean((sys.H(z.clone().detach()) - net(Variable(z.clone()).to(device).float()).detach().cpu()[:,0])**2),
                "time":float(details[-1][:-3])})           
            df = pd.concat([df,data.to_frame().T], ignore_index = True)
        elif (len(details) == 5) & (details[0] == "SepNet4") & (int(details[3])>0):
            net = sepNet4(sys.sepshape1,sys.sepshape2,sys.sepshape3,1) 
            net.load_state_dict(torch.load(f))
            model = "sHNN-I4"
            net = net.to(device)
            data = pd.Series({'seed':int(details[1]), 
                'ini':int(details[2]), 'model':model, 'epochs':int(details[3]),
                'loss':float(error(fvec(z), np.concatenate(get_grad(net, z, device)))/(n_sample**dim)), 
                'Hloss':torch.mean((sys.H(z.clone().detach()) - net(Variable(z.clone()).to(device).float()).detach().cpu()[:,0])**2),
                "time":float(details[-1][:-3])})            
            df = pd.concat([df,data.to_frame().T], ignore_index = True)
        elif (len(details) == 5) & (details[0] == "SepNet5") & (int(details[3])>0):
            net = sepNet5(sys.sepshape1,sys.sepshape2,sys.sepshape3,1) 
            net.load_state_dict(torch.load(f))
            model = "sHNN-I5"
            net = net.to(device)
            data = pd.Series({'seed':int(details[1]), 
                'ini':int(details[2]), 'model':model, 'epochs':int(details[3]),
                'loss':float(error(fvec(z), np.concatenate(get_grad(net, z, device)))/(n_sample**dim)), 
                'Hloss':torch.mean((sys.H(z.clone().detach()) - net(Variable(z.clone()).to(device).float()).detach().cpu()[:,0])**2),
                "time":float(details[-1][:-3])})            
            df = pd.concat([df,data.to_frame().T], ignore_index = True)


    for f in glob.glob(loc+"/Experiments/Results/%s/*.pt" %s) :
        details = f.split("/")[-1].split("_")
        if (len(details) == 5) & (details[0] == "sepNN") :
            net = sepNet(sys.sepshape1,sys.sepshape2,sys.sepshape3,1) 
            net.load_state_dict(torch.load(f))
            model = "sHNN-I"
            net = net.to(device)
            data = pd.Series({'seed':int(details[1]), 
                'ini':int(details[2]), 'model':model, 'epochs':int(details[3]),
                'loss':float(error(fvec(z), np.concatenate(get_grad(net, z, device)))/(n_sample**dim)), 
                'Hloss':torch.mean((sys.H(z.clone().detach()) - net(Variable(z.clone()).to(device).float()).detach().cpu()[:,0])**2),
                "time":float(details[-1][:-3])})             
            if int(details[3])>0:
              df = pd.concat([df,data.to_frame().T], ignore_index = True)
    
    df["timeoriginal"] = df["time"]
    # df["tpe"] = df["time"]/df["epochs"]
    df = df.drop(['timeoriginal'], axis=1)
    df1 = df.groupby(by = ["ini","model"], as_index = False).agg(['mean','sem'])
    df1 = df1.reset_index()
    df1 = df1[df1["ini"]==512]
    print(df1)
    

    print("%.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E)"
								%(df1[df1["model"]=="sHNN-I"]["loss"]["mean"].values, df1[df1["model"]=="sHNN-I"]["loss"]["sem"].values, 
								df1[df1["model"]=="sHNN-I1"]["loss"]["mean"].values, df1[df1["model"]=="sHNN-I1"]["loss"]["sem"].values, 
    								df1[df1["model"]=="sHNN-I2"]["loss"]["mean"].values, df1[df1["model"]=="sHNN-I2"]["loss"]["sem"].values, 
								df1[df1["model"]=="sHNN-I3"]["loss"]["mean"].values, df1[df1["model"]=="sHNN-I3"]["loss"]["sem"].values, 
								df1[df1["model"]=="sHNN-I4"]["loss"]["mean"].values, df1[df1["model"]=="sHNN-I4"]["loss"]["sem"].values, 
								df1[df1["model"]=="sHNN-I5"]["loss"]["mean"].values, df1[df1["model"]=="sHNN-I5"]["loss"]["sem"].values, ))

    print("%.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E)"
								%(df1[df1["model"]=="sHNN-I"]["Hloss"]["mean"].values, df1[df1["model"]=="sHNN-I"]["Hloss"]["sem"].values, 
								df1[df1["model"]=="sHNN-I1"]["Hloss"]["mean"].values, df1[df1["model"]=="sHNN-I1"]["Hloss"]["sem"].values, 
    								df1[df1["model"]=="sHNN-I2"]["Hloss"]["mean"].values, df1[df1["model"]=="sHNN-I2"]["Hloss"]["sem"].values, 
								df1[df1["model"]=="sHNN-I3"]["Hloss"]["mean"].values, df1[df1["model"]=="sHNN-I3"]["Hloss"]["sem"].values, 
								df1[df1["model"]=="sHNN-I4"]["Hloss"]["mean"].values, df1[df1["model"]=="sHNN-I4"]["Hloss"]["sem"].values, 
								df1[df1["model"]=="sHNN-I5"]["Hloss"]["mean"].values, df1[df1["model"]=="sHNN-I5"]["Hloss"]["sem"].values, ))

    print("%.2f (%.0f) & %.2f (%.0f) & %.2f (%.0f) & %.2f (%.0f) & %.2f (%.0f) & %.2f (%.0f)"
								%(df1[df1["model"]=="sHNN-I"]["time"]["mean"].values, df1[df1["model"]=="sHNN-I"]["epochs"]["mean"].values, 
								df1[df1["model"]=="sHNN-I1"]["time"]["mean"].values, df1[df1["model"]=="sHNN-I1"]["epochs"]["mean"].values, 
    								df1[df1["model"]=="sHNN-I2"]["time"]["mean"].values, df1[df1["model"]=="sHNN-I2"]["epochs"]["mean"].values, 
								df1[df1["model"]=="sHNN-I3"]["time"]["mean"].values, df1[df1["model"]=="sHNN-I3"]["epochs"]["mean"].values, 
								df1[df1["model"]=="sHNN-I4"]["time"]["mean"].values, df1[df1["model"]=="sHNN-I4"]["epochs"]["mean"].values, 
								df1[df1["model"]=="sHNN-I5"]["time"]["mean"].values, df1[df1["model"]=="sHNN-I5"]["epochs"]["mean"].values, ))


