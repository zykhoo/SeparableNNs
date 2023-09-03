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
import matplotlib


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


def is_pareto_efficient(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


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

for s in ["pendulum","trigo","arctan","logarithm","anisotropicoscillator2D","henonheiles","todalattice","coupledoscillator"]:
    exec("sys = all_systems.%s" %s)
    dim = len(sys.spacedim)
    df = pd.DataFrame(columns = ["seed","ini","model","loss","epochs","time","x"])
    z = torch.tensor(np.array(np.meshgrid(*[np.linspace(sys.spacedim[i][0], sys.spacedim[i][1], 10) for i in range(dim)],
                                      ))).reshape(len(sys.spacedim), 10**dim).transpose(1,0)
    for f in glob.glob(loc+"/Experiments/ResultsObservational/%s/*.pt" %s):
        details = f.split("/")[-1].split("_")
        fvec = lambda z: np.concatenate([np.expand_dims(sys.f1(z),0), np.expand_dims(sys.f2(z),0)]) if len(sys.spacedim)==2 else np.concatenate([sys.f1(z), sys.f2(z)])
        if (len(details) == 5) & (details[0][:6] == "obsNet"):
            net = Net(sys.netshape1,sys.netshape2,1)
            net.load_state_dict(torch.load(f))
            model = "sHNN-O"
            net = net.to(device)
            data = pd.Series({'seed':int(details[1]), 
                'ini':int(details[2]), 'model':model, 'epochs':int(details[3]),
                'loss':float(error(fvec(z), np.concatenate(get_grad(net, z, device)))/(n_sample**dim)), 
                'Hloss':torch.mean((sys.H(z.clone().detach()) - net(Variable(z.clone()).to(device).float()).detach().cpu()[:,0])**2),
                'x': int(details[0][6:]),
                "time":float(details[-1][:-3])})         
            if  (int(details[3])>0):
              df = pd.concat([df,data.to_frame().T], ignore_index = True)

    df = df.groupby(by = ["ini","x","model"], as_index = False).agg(['mean','sem'])
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df = df.reset_index()
    df = df[df["ini"]==512]
    df = df[["x","seed_mean","loss_mean","loss_sem","Hloss_mean","Hloss_sem","time_mean","epochs_mean"]]

    df["invloss"] = 1/df["loss_mean"]
    df["invtime"] = 1/df["time_mean"]
    print(df)

    print("%s & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E)" %(s, df[df["x"]==1]["loss_mean"], df[df["x"]==1]["loss_sem"], 
								df[df["x"]==2]["loss_mean"], df[df["x"]==2]["loss_sem"],
								df[df["x"]==3]["loss_mean"], df[df["x"]==3]["loss_sem"],
								df[df["x"]==4]["loss_mean"], df[df["x"]==4]["loss_sem"],
								df[df["x"]==5]["loss_mean"], df[df["x"]==5]["loss_sem"],))
#    print("%s & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E)" %(s, df[df["x"]==1]["Hloss_mean"], df[df["x"]==1]["Hloss_sem"], 
#								df[df["x"]==2]["Hloss_mean"], df[df["x"]==2]["Hloss_sem"],
#								df[df["x"]==3]["Hloss_mean"], df[df["x"]==3]["Hloss_sem"],
#								df[df["x"]==4]["Hloss_mean"], df[df["x"]==4]["Hloss_sem"],
#								df[df["x"]==5]["Hloss_mean"], df[df["x"]==5]["Hloss_sem"],))
    print("%s & %.2f (%.0f) & %.2f (%.0f) & %.2f (%.0f) & %.2f (%.0f) & %.2f (%.0f)" %(s, df[df["x"]==1]["time_mean"].values, df[df["x"]==1]["epochs_mean"].values[0], 
								df[df["x"]==2]["time_mean"].values, df[df["x"]==2]["epochs_mean"].values[0],
								df[df["x"]==3]["time_mean"].values, df[df["x"]==3]["epochs_mean"].values[0],
								df[df["x"]==4]["time_mean"].values, df[df["x"]==4]["epochs_mean"].values[0],
								df[df["x"]==5]["time_mean"].values, df[df["x"]==5]["epochs_mean"].values[0],))

    print("%s & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E)" %(s, df[df["x"]==10]["loss_mean"], df[df["x"]==10]["loss_sem"], 
								df[df["x"]==20]["loss_mean"], df[df["x"]==20]["loss_sem"],
								df[df["x"]==30]["loss_mean"], df[df["x"]==30]["loss_sem"],
								df[df["x"]==40]["loss_mean"], df[df["x"]==40]["loss_sem"],
								df[df["x"]==50]["loss_mean"], df[df["x"]==50]["loss_sem"],))
#    print("%s & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E)" %(s, df[df["x"]==10]["Hloss_mean"], df[df["x"]==10]["Hloss_sem"], 
#								df[df["x"]==20]["Hloss_mean"], df[df["x"]==20]["Hloss_sem"],
#								df[df["x"]==30]["Hloss_mean"], df[df["x"]==30]["Hloss_sem"],
#								df[df["x"]==40]["Hloss_mean"], df[df["x"]==40]["Hloss_sem"],
#								df[df["x"]==50]["Hloss_mean"], df[df["x"]==50]["Hloss_sem"],))
    print("%s & %.2f (%.0f) & %.2f (%.0f) & %.2f (%.0f) & %.2f (%.0f) & %.2f (%.0f)" %(s, df[df["x"]==10]["time_mean"].values, df[df["x"]==10]["epochs_mean"].values[0], 
								df[df["x"]==20]["time_mean"].values, df[df["x"]==20]["epochs_mean"].values[0],
								df[df["x"]==30]["time_mean"].values, df[df["x"]==30]["epochs_mean"].values[0],
								df[df["x"]==40]["time_mean"].values, df[df["x"]==40]["epochs_mean"].values[0],
								df[df["x"]==50]["time_mean"].values, df[df["x"]==50]["epochs_mean"].values[0],))

    print("%s & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E)" %(s, df[df["x"]==100]["loss_mean"], df[df["x"]==100]["loss_sem"], 
								df[df["x"]==200]["loss_mean"], df[df["x"]==200]["loss_sem"],
								df[df["x"]==300]["loss_mean"], df[df["x"]==300]["loss_sem"],
								df[df["x"]==400]["loss_mean"], df[df["x"]==400]["loss_sem"],))
#    print("%s & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E)" %(s, df[df["x"]==100]["Hloss_mean"], df[df["x"]==100]["Hloss_sem"], 
#								df[df["x"]==200]["Hloss_mean"], df[df["x"]==200]["Hloss_sem"],
#								df[df["x"]==300]["Hloss_mean"], df[df["x"]==300]["Hloss_sem"],
#								df[df["x"]==400]["Hloss_mean"], df[df["x"]==400]["Hloss_sem"],))
    print("%s & %.2f (%.0f) & %.2f (%.0f) & %.2f (%.0f) & %.2f (%.0f)" %(s, df[df["x"]==100]["time_mean"].values, df[df["x"]==100]["epochs_mean"].values[0], 
								df[df["x"]==200]["time_mean"].values, df[df["x"]==200]["epochs_mean"].values[0],
								df[df["x"]==300]["time_mean"].values, df[df["x"]==300]["epochs_mean"].values[0],
								df[df["x"]==400]["time_mean"].values, df[df["x"]==400]["epochs_mean"].values[0],))



    
    if s == "todalattice":
      df = df[df.x != 400]
    """
    costs = df[["loss_mean", "time_mean"]].to_numpy()
    optimal = df["x"][is_pareto_efficient(costs)].values
    fig, ax = plt.subplots()
    ax.scatter(df["invloss"], df["invtime"])
    for i, txt in enumerate(df["x"].values):
        ax.annotate(txt, (df["invloss"][i], df["invtime"][i]))
    ax.scatter(df[df["x"].isin(optimal)]["invloss"], df[df["x"].isin(optimal)]["invtime"])
    plt.savefig(loc+"/images/ResultsObservational/invpareto_%s.jpg" %s)
    fig, ax = plt.subplots()
    ax.scatter(df["loss_mean"], df["time_mean"])
    for i, txt in enumerate(df["x"].values):
        ax.annotate(txt, (df["loss_mean"][i], df["time_mean"][i]))
    ax.scatter(df[df["x"].isin(optimal)]["loss_mean"], df[df["x"].isin(optimal)]["time_mean"])
    plt.savefig(loc+"/images/ResultsObservational/pareto_%s.jpg" %s)
    plt.close()
    """
    
    matplotlib.rc('font', size=16)
    df['x'] = df["x"].astype('string')
    fig, ax = plt.subplots(layout='constrained')
    ax.plot(df['x'], df['loss_mean'], label = "vector error", color = "blue")
    axr = ax.twinx()
    axr.plot([], [], color = 'blue', label = 'vector error')
    axr.plot(df['x'], df['time_mean'], label = "training time", linestyle = "--", color = "orange")
    ax.tick_params(axis = 'y', labelsize=16)
    ax.tick_params(axis = 'x', rotation = -30, labelsize=16)
    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
    axr.tick_params(axis = 'y', labelsize=16)
    axr.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
    plt.legend(fontsize=20, loc='upper right')
    plt.savefig(loc+"/images/ResultsObservational/%s.jpg" %s, dpi = 600)
    plt.close()

    


