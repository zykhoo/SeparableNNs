import os 
import sys
from pathlib import Path

sys.path.insert(0, '/home/ziyu/SeparableHNN')
import all_systems

loc = os.getcwd()

import time 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.nn.utils.prune as prune
from torch.autograd import Variable
from tqdm import tqdm
import math
import numpy as np
from sklearn.model_selection import train_test_split
import glob
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import random
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.set_default_dtype(torch.float64)

deviceno = int(0)
if torch.cuda.is_available():
  device=torch.device('cuda:%s' %deviceno)
else:
  device=torch.device('cpu')

s = "todalattice"
exec("sys = all_systems.%s" %s)

c = '%sCheckpoint.pt' %s

initialcon = [512]


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

def lossfuc(model,mat,x,y,device,x0,H0,dim,c1=1,c2=1,c3=1,c4=1,verbose=False):
    dim = int(wholemat.shape[1]/2)
    f3=(model(torch.tensor([[x0]*dim]).to(device))-torch.tensor([[H0]]).to(device))**2
    dH=torch.autograd.grad(y, x, grad_outputs=y.data.new(y.shape).fill_(1),retain_graph=True,create_graph=True, allow_unused=True)[0]
    dHdq=dH[:,:int(dim/2)]
    d2Hdqp = torch.zeros(1, dHdq.shape[1]).to(device)
    if c4 != 0:
      for i in range(dHdq.shape[1]): # [0] is batch, [1] is value of gradient
        # print(torch.unsqueeze(dHdq[:,i],1).shape, y.shape)
        d2Hdqp += torch.mean(torch.abs(torch.autograd.grad(torch.unsqueeze(dHdq[:,i],1), x, grad_outputs=torch.unsqueeze(dHdq[:,i],1).data.new(torch.unsqueeze(dHdq[:,i],1).shape).fill_(1),create_graph=True, allow_unused=True)[0][:,int(dim/2):]), dim=0)
    d2Hdqp = (d2Hdqp**2) / (dim/2)
    dHdp=dH[:,int(dim/2):]
    qprime=(mat[:,dim:int(3*dim/2)])
    pprime=(mat[:,int(3*dim/2):])
    assert dHdq.shape[1] == int(dim/2)
    assert dHdp.shape[1] == int(dim/2)
    assert qprime.shape[1] == int(dim/2)
    assert pprime.shape[1] == int(dim/2)
    f1=torch.mean((dHdp-qprime)**2,dim=0)
    # print(dHdq, pprime)
    f2=torch.mean((dHdq+pprime)**2,dim=0)
    f4=d2Hdqp #torch.mean((dHdq*qprime+dHdp*pprime)**2,dim=0)
    loss=torch.mean(c1*f1+c2*f2+c3*f3+c4*f4)
    # if loss > 1000: print("errors:", f1, f2, f3, f4)
    meanf1,meanf2,meanf3,meanf4=torch.mean(c1*f1),torch.mean(c2*f2),torch.mean(c3*f3),torch.mean(c4*f4)
    if verbose:
      print(x)
      print(meanf1,meanf2,meanf3,meanf4)
      print(loss,meanf1,meanf2,meanf3,meanf4)
    return loss,meanf1,meanf2,meanf3,meanf4


def data_preprocessing(start_train, final_train,device):       
    wholemat = np.hstack((start_train.transpose(), final_train.transpose()))
    wholemat = torch.tensor(wholemat)
    wholemat = wholemat.to(device)
    wholemat,evalmat=train_test_split(wholemat, train_size=0.8, random_state=1)
    return wholemat,evalmat


# evaluate loss of dataset 
def get_loss(model,device,initial_conditions,bs,x0,H0,dim,wholemat,evalmat,c1,c2,c3,c4,trainset=False,verbose=False):
    # this function is used to calculate average loss of a whole dataset
    # rootpath: path of set to be calculated loss
    # model: model
    # trainset: is training set or not
    if trainset:
        mat=wholemat
    else:
        mat=evalmat
    avg_loss=0
    avg_f1=0
    avg_f2=0
    avg_f3=0
    avg_f4=0
    for count in range(0,len(mat),bs):
      curmat=mat[count:count+bs]
      x=Variable((curmat[:,:dim]).to(torch.float64),requires_grad=True)
      y=model(x)
      x=x.to(device)
      loss,f1,f2,f3,f4=lossfuc(model,curmat,x,y,device,x0,H0,dim,c1,c2,c3,c4)
      avg_loss+=loss.detach().cpu().item()
      avg_f1+=f1.detach().cpu().item()
      avg_f2+=f2.detach().cpu().item()
      avg_f3+=f3.detach().cpu().item()
      avg_f4+=f4.detach().cpu().item()
    num_batches=len(mat)//bs
    avg_loss/=num_batches
    avg_f1/=num_batches
    avg_f2/=num_batches
    avg_f3/=num_batches
    avg_f4/=num_batches
    if verbose:
        print(' loss=',avg_loss,' f1=',avg_f1,' f2=',avg_f2,' f3=',avg_f3,' f4=',avg_f4)
    return avg_loss


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if abs(self.counter-self.patience)<5:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), c)
        self.val_loss_min = val_loss

def train(net,name,bs,num_epoch,initial_conditions,device,wholemat,evalmat,x0,H0,dim,LR,patience,c1,c2,c3,c4):
    starttime = time.time() 
    # function of training process
    # net: the model
    # bs: batch size 
    # num_epoch: max of epoch to run
    # initial_conditions: number of trajectory in train set
    # patience: EarlyStopping parameter
    # c1~c4: hyperparameter for loss function

    smarker = 1
    avg_lossli,avg_f1li,avg_f2li,avg_f3li,avg_f4li=[],[],[],[],[]
    avg_vallosses=[]
    
    start = time.time()
    lr = LR # initial learning rate
    net=net.to(device)


    early_stopping = EarlyStopping(patience=patience, verbose=False,delta=0.00001) # delta
    optimizer=torch.optim.Adam(net.parameters() , lr=LR )

    for epoch in range(num_epoch):

        running_loss=0

        running_f1=0
        running_f2=0
        running_f3=0
        running_f4=0
        num_batches=0
        
        # train
        shuffled_indices=torch.linspace(0,len(wholemat)-1,len(wholemat)).type(torch.long)
        net.train()
        for count in range(0,len(wholemat),bs):
            optimizer.zero_grad()

            indices=shuffled_indices[count:count+bs]
            mat=wholemat[indices]

            x=Variable(torch.tensor(mat[:,:dim]).to(torch.float64),requires_grad=True)
            y=net(x)

            loss,f1,f2,f3,f4=lossfuc(net,mat,x,y,device,x0,H0,dim,c1,c2,c3,c4)  
            loss.backward()
            torch.nn.utils.clip_grad_norm(net.parameters(), 1)

            optimizer.step()

            # compute some stats
            running_loss += loss.detach().item()
            running_f1 += f1.detach().item()
            running_f2 += f2.detach().item()
            running_f3 += f3.detach().item()
            running_f4 += f4.detach().item()

            num_batches+=1
            torch.cuda.empty_cache()



        avg_loss = running_loss/num_batches
        avg_f1 = running_f1/num_batches
        avg_f2 = running_f2/num_batches
        avg_f3 = running_f3/num_batches
        avg_f4 = running_f4/num_batches
        elapsed_time = time.time() - start
        
        avg_lossli.append(avg_loss)
        avg_f1li.append(avg_f1)
        avg_f2li.append(avg_f2)
        avg_f3li.append(avg_f3)
        avg_f4li.append(avg_f4)
        
        
        # evaluate
        net.eval()
        avg_val_loss=get_loss(net,device,len(evalmat),bs,x0,H0,dim,wholemat,evalmat,c1,c2,c3,c4)
        avg_vallosses.append(avg_val_loss)
        
        if epoch % 500 == 0 : 
            # print(' ')
            print('epoch=',epoch, ' time=', elapsed_time,
                  ' loss=', avg_loss ,' val_loss=',avg_val_loss,' f1=', avg_f1 ,' f2=', avg_f2 ,
                  ' f3=', avg_f3 ,' f4=', avg_f4 , 'num_batches=', num_batches, 'percent lr=', optimizer.param_groups[0]["lr"] )

        if time.time() - starttime > smarker:
            torch.save(net.state_dict(), "%s_%s_%s.pt" %(name,epoch,time.time()-starttime))
            smarker += 20
        
        if epoch%100 == 0:
            torch.save(net.state_dict(), c)
        
        if math.isnan(running_loss):
            text_file = open("nan_report.txt", "w")
            text_file.write('name=%s at epoch %s' %(name, epoch))
            text_file.close()
            print("saving this file and ending the training")
            net.load_state_dict(torch.load(c)) 
            return net,epoch,avg_vallosses,avg_lossli,avg_f1li,avg_f2li,avg_f3li,avg_f4li

        
        
        early_stopping(avg_val_loss,net)
        if early_stopping.early_stop:
            print('Early Stopping')
            break
            
    net.load_state_dict(torch.load(c)) #net=torch.load(c)
    return net,epoch,avg_vallosses,avg_lossli,avg_f1li,avg_f2li,avg_f3li,avg_f4li

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
        # print("input", torch.sum(x, (1,2)))
        # print("initial", x.shape)
        x = softplus(self.hidden_layer_1(x)) 
        # print("first", torch.sum(x, (1,2)))
        # print("hl1", x.shape)
        x = softplus(self.hidden_layer_2(x)) 
        # print("second", torch.sum(x, (1,2)))
        # print("hl2", x.shape)
        x = self.output_layer(x)
        # print("third", torch.sum(x, (1,2)))
        # print("output", x.shape)
        x = torch.sum(x, dim = 0)
        # print("out", torch.sum(x))
        return x


def CreateTrainingDataExact(traj_len,ini_con,spacedim,h,f1,f2,seed,n_h = 800,t=None):
  np.random.seed(seed = seed)
  start = np.vstack([np.random.uniform(low = spacedim[i][0], high = spacedim[i][1], size = ini_con) for i in range(len(spacedim))])
  f = lambda x: np.expand_dims(np.hstack([f1(x), f2(x)]),1) 
  delta = f(start[:,0])
  for k in range(ini_con-1):
    new_delta = f(np.squeeze(start[:,k+1]))
    delta = np.hstack((delta, new_delta))
  return start, delta


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


def weights_copy(HetNet, sepnet):
  with torch.no_grad():	
    HetNet.hidden_layer_1a.weight.copy_(sepnet.hidden_layer_1.weights[0])
    HetNet.hidden_layer_2a.weight.copy_(sepnet.hidden_layer_1.weights[1])
    HetNet.hidden_layer_1a.bias.copy_(sepnet.hidden_layer_1.bias[0,0,:])
    HetNet.hidden_layer_2a.bias.copy_(sepnet.hidden_layer_1.bias[1,0,:])
    HetNet.hidden_layer_1b.weight.copy_(sepnet.hidden_layer_2.weights[0])
    HetNet.hidden_layer_2b.weight.copy_(sepnet.hidden_layer_2.weights[1])
    HetNet.hidden_layer_1b.bias.copy_(sepnet.hidden_layer_2.bias[0,0,:])
    HetNet.hidden_layer_2b.bias.copy_(sepnet.hidden_layer_2.bias[1,0,:])
    HetNet.hidden_layer_1c.weight.copy_(sepnet.output_layer.weights[0])
    HetNet.hidden_layer_2c.weight.copy_(sepnet.output_layer.weights[1])
    HetNet.hidden_layer_1c.bias.copy_(sepnet.output_layer.bias[0,0,:])
    HetNet.hidden_layer_2c.bias.copy_(sepnet.output_layer.bias[1,0,:])
  return HetNet



for i in range(20):
  seed = i
  np.random.seed(seed=seed)
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  for ini in initialcon: 

    x0, H0, LR, h, f1, f2, dim = sys.x0, sys.H0, sys.LR, sys.h, sys.f1gen, sys.f2gen, len(sys.spacedim)


    #start, delta = CreateTrainingDataExact(1,ini,sys.spacedim,h,f1,f2,seed = seed,n_h = 1,t=None)
    

    # wholemat, evalmat = data_preprocessing(start, delta, device) 
    wholemat = torch.tensor(np.loadtxt(str(Path(loc).parents[0]) + '/Results/data/%s_%s_wholemat.txt' %(s,seed))).to(device)
    evalmat = torch.tensor(np.loadtxt(str(Path(loc).parents[0]) + '/Results/data/%s_%s_evalmat.txt' %(s,seed))).to(device)
 
    expmat = wholemat
    

    obswholemat = torch.vstack((torch.stack(([wholemat[(torch.linspace(1, len(wholemat), steps = len(wholemat))%(len(wholemat))).type(torch.long),d] for d in range(int(dim/2))])),
                                torch.stack(([wholemat[:,d] for d in range(int(dim/2),dim)])),
                                torch.stack([wholemat[:,d] for d in range(dim, dim+int(dim/2))]),
                                torch.stack([wholemat[(torch.linspace(1, len(wholemat), steps = len(wholemat))%(len(wholemat))).type(torch.long),d] for d in range(dim+int(dim/2),2*dim)]) ))
    print(wholemat)

    sepnet0 = sepNet(sys.sepshape1,sys.sepshape2,sys.sepshape3,sys.sepshape4) 
    sepnet = sepNet(sys.sepshape1,sys.sepshape2,sys.sepshape3,sys.sepshape4) 
    sepnet0.load_state_dict(torch.load(str(Path(loc).parents[0]) + '/Results/%s/%s_%s_%s_%s_%s.pt' %(s,"sepNN",seed,ini,0,0)))
    sepnet.load_state_dict(torch.load(str(Path(loc).parents[0]) + '/Results/%s/%s_%s_%s_%s_%s.pt' %(s,"sepNN",seed,ini,0,0)))
    for name, param in sepnet0.named_parameters():
      print(name, torch.sum(param.data), param.data.shape)
    starttime = time.time() 
    torch.save(sepnet.state_dict(), '%s/%s/%s_%s_%s_%s_%s.pt' %(loc,s,"SepNet",seed,ini,0,0))
    results = train(sepnet, name="%s/%s/%s_%s_SepNet" %(loc,s,seed,ini),bs=int(len(wholemat)/5),num_epoch=150001,initial_conditions=initialcon,device=device, wholemat=wholemat,evalmat=evalmat,x0=x0,H0=H0,dim=dim,LR=LR,patience=4000,c1=1,c2=1,c3=1,c4=0)
    sepnet, epochs = results[0], results[1]
    septraintime = time.time()-starttime
    torch.save(sepnet.state_dict(), '%s/%s/%s_%s_%s_%s_%s.pt' %(loc,s,"SepNet",seed,ini,epochs,septraintime))

    HetNet = HsepNet(sys.sepshape1,sys.sepshape2,sys.sepshape3,sys.sepshape4) 
    HetNet = weights_copy(HetNet, sepnet0)
    for name, param in HetNet.named_parameters():
      print(name, torch.sum(param.data), param.data.shape)
    torch.save(HetNet.state_dict(), '%s/%s/%s_%s_%s_%s_%s.pt' %(loc,s,"HetNet",seed,ini,0,0))
    HetNet.load_state_dict(torch.load('%s/%s/%s_%s_%s_%s_%s.pt' %(loc,s,"HetNet",seed,ini,0,0)))
    starttime = time.time() 
    results = train(HetNet, name="%s/%s/%s_%s_HetNet" %(loc,s,seed,ini),bs=int(len(wholemat)/5),num_epoch=150001,initial_conditions=initialcon,device=device, wholemat=wholemat,evalmat=evalmat,x0=x0,H0=H0,dim=dim,LR=LR,patience=4000,c1=1,c2=1,c3=1,c4=0)
    HetNet, epochs = results[0], results[1]
    septraintime = time.time()-starttime
    torch.save(HetNet.state_dict(), '%s/%s/%s_%s_%s_%s_%s.pt' %(loc,s,"HetNet",seed,ini,epochs,septraintime))

