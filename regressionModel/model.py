# -*- coding: utf-8 -*-
### Using pytorch exapmle from http://pytorch.org/tutorials/beginner/pytorch_with_examples.html ###

#TODO: solve GPU Issu
###     Found GPU0 GeForce 940M which is of cuda capability 5.0.
###    PyTorch no longer supports this GPU because it is too old.



import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import time
from datetime import datetime
from pathlib import Path

my_file = Path("model.pt")

from torch.nn.modules.module import Module

class CAPELoss(Module):
    def __init__(self):
        super(CAPELoss, self).__init__()

    def forward(self, input, target):
        basis = 122400.0
        return (input - target).abs().sum()/(len(input) * basis)




pandaData = pd.read_csv("../Aufgabenstellung/train.csv")
numpyData = pandaData.values
#convert timestamp to float
numpyData[:,0] = list(map(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')),numpyData[:,0]))
#Testing:
# numpyData = numpyData[:100] #TODO: comment!

x = numpyData[:,:-1].astype(float)
y = numpyData[:,-1].astype(float)


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N = numpyData.shape[0]
D_in = x.shape[1]
D_out = 1
H = 100     #TODO: Maybe more and/or larger hidden layer

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU



# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
x = Variable(torch.from_numpy(x).type(dtype))
y = Variable(torch.from_numpy(y).type(dtype), requires_grad=False)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)


#Define loss_fn like this http://pytorch.org/docs/master/notes/extending.html
#https://discuss.pytorch.org/t/solved-what-is-the-correct-way-to-implement-custom-loss-function/3568/4
def loss_fn(y_pred, y):
    basis = 122400.0
    dif = (y_pred.data - y.data).abs()
    retval = dif.sum()/(len(dif) * basis)
    return Variable(torch.FloatTensor([retval]), requires_grad = True)
    # basis = 122400.0
    # dif = np.absolute(y_pred.data.numpy() - y.data.numpy())
    # retval = np.sum(dif) / (len(dif) * basis)
    # print (retval)
    # return Variable(torch.from_numpy(retval).type(dtype))

# loss_fn = torch.nn.MSELoss(size_average=False) #TODO: Use correct lossfunction
loss_fn = CAPELoss()

if my_file.is_file():
    model = torch.load(my_file)

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Variables it should update.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    # print("loss.requires_grad: {}".format(loss.requires_grad))
    # print("loss.grad: {}".format(loss.grad))
    # print("loss.is_leaf: {}".format(loss.is_leaf))
    # print("loss.grad_fn: {}".format(loss.grad_fn))

    print(t, loss.data[0])
    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

torch.save(model, my_file)
print("saved model to {}".format(my_file))
