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
    for f in glob.glob(loc+"/Experiments/ResultsLearningCoefficient/%s/*.pt" %s):
        details = f.split("/")[-1].split("_")
        fvec = lambda z: np.concatenate([np.expand_dims(sys.f1(z),0), np.expand_dims(sys.f2(z),0)]) if len(sys.spacedim)==2 else np.concatenate([sys.f1(z), sys.f2(z)])
        if (len(details) == 6) & (str(details[0]) == "PINN"): # its a softsep PINN
            net = Net(sys.netshape1,sys.netshape2,1)
            net.load_state_dict(torch.load(f))
            model = "sHNN-L"
            net = net.to(device)
            data = pd.Series({'seed':int(details[2]), 'x':str(details[1][-4:]),
                'ini':int(details[3]), 'model':model, 'epochs':int(details[4]),
                'loss':float(error(fvec(z), np.concatenate(get_grad(net, z, device)))/(n_sample**dim)), 
                'Hloss':torch.mean((sys.H(z.clone().detach()) - net(Variable(z.clone()).to(device).float()).detach().cpu()[:,0])**2),
                "time":float(details[-1][:-3])})
            if  (int(details[4])>0):
              df = pd.concat([df,data.to_frame().T], ignore_index = True)

    df = df.groupby(by = ["ini","x","model"], as_index = False).agg(['mean','sem'])
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df = df.reset_index()
    df = df[df["ini"]==512]
    df = df[["x","loss_mean","loss_sem","Hloss_mean","Hloss_sem","time_mean","epochs_mean"]]
    print(df)

    print("%s & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E)" %(s, df[df["x"]=="0.25"]["loss_mean"], df[df["x"]=="0.25"]["loss_sem"], 
								df[df["x"]=="0.50"]["loss_mean"], df[df["x"]=="0.50"]["loss_sem"],
								df[df["x"]=="1.00"]["loss_mean"], df[df["x"]=="1.00"]["loss_sem"],
								df[df["x"]=="2.00"]["loss_mean"], df[df["x"]=="2.00"]["loss_sem"],
								df[df["x"]=="4.00"]["loss_mean"], df[df["x"]=="4.00"]["loss_sem"],))
    print("%s & 0.00 \\%% & %.2f \\%% & %.2f \\%% & %.2f \\%% & %.2f \\%%" %(s, (1-df[df["x"]=="0.25"]["loss_mean"].values/df[df["x"]=="1.00"]["loss_mean"].values)*100, 
								(1-df[df["x"]=="0.50"]["loss_mean"].values/df[df["x"]=="1.00"]["loss_mean"].values)*100, 
								(1-df[df["x"]=="2.00"]["loss_mean"].values/df[df["x"]=="1.00"]["loss_mean"].values)*100,
								(1-df[df["x"]=="4.00"]["loss_mean"].values/df[df["x"]=="1.00"]["loss_mean"].values)*100,))
    print("%s & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E)" %(s, df[df["x"]=="0.25"]["Hloss_mean"], df[df["x"]=="0.25"]["Hloss_sem"], 
								df[df["x"]=="0.50"]["Hloss_mean"], df[df["x"]=="0.50"]["Hloss_sem"],
								df[df["x"]=="1.00"]["Hloss_mean"], df[df["x"]=="1.00"]["Hloss_sem"],
								df[df["x"]=="2.00"]["Hloss_mean"], df[df["x"]=="2.00"]["Hloss_sem"],
								df[df["x"]=="4.00"]["Hloss_mean"], df[df["x"]=="4.00"]["Hloss_sem"],))
    print("%s & 0.00 \\%% & %.2f \\%% & %.2f \\%% & %.2f \\%% & %.2f \\%%" %(s, (1-df[df["x"]=="0.25"]["Hloss_mean"].values/df[df["x"]=="1.00"]["Hloss_mean"].values)*100, 
								(1-df[df["x"]=="0.50"]["Hloss_mean"].values/df[df["x"]=="1.00"]["Hloss_mean"].values)*100, 
								(1-df[df["x"]=="2.00"]["Hloss_mean"].values/df[df["x"]=="1.00"]["Hloss_mean"].values)*100,
								(1-df[df["x"]=="4.00"]["Hloss_mean"].values/df[df["x"]=="1.00"]["Hloss_mean"].values)*100,))

	
    print("%s & %.2f (%.0f) & %.2f (%.0f) & %.2f (%.0f) & %.2f (%.0f) & %.2f (%.0f)" %(s, df[df["x"]=="0.25"]["time_mean"].values, df[df["x"]=="0.25"]["epochs_mean"].values[0], 
								df[df["x"]=="0.50"]["time_mean"].values, df[df["x"]=="0.50"]["epochs_mean"].values[0],
								df[df["x"]=="1.00"]["time_mean"].values, df[df["x"]=="1.00"]["epochs_mean"].values[0],
								df[df["x"]=="2.00"]["time_mean"].values, df[df["x"]=="2.00"]["epochs_mean"].values[0],
								df[df["x"]=="4.00"]["time_mean"].values, df[df["x"]=="4.00"]["epochs_mean"].values[0],))




