import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from random import randint
from numpy import arange
from numpy import meshgrid
from matplotlib import pyplot

#analytic solution to the equation (true solution)
def analytic(x,y):
    return math.e**(-x)*(x+y**3)

#generate training data
X = []
for i in range(1,10):
    i=i*0.1
    for j in range(1,10):
        j = j*0.1
        X.append([i,j])

#generate testing data
x_0=[]
for i in range(0,101):
    i=i*0.01
    for j in range(0,101):
        j=j*0.01
        x_0.append([i,j])
x_0 = torch.tensor(x_0, dtype=torch.float32)

X = torch.tensor(X, dtype=torch.float32)

#our network architecture
model = nn.Sequential(
nn.Linear(2, 10),
nn.Sigmoid(),
nn.Linear(10, 1,bias=False),
)

optimizer = optim.Adam(model.parameters(), lr=0.001) #we use the ADAM optimisation method
exp = math.e

n_epochs = 100
for epoch in range(n_epochs):
    A = torch.clone(X) #copy training set X, since we will be updating this new set A while training (don't want to change the actual training set though)
    for i in range(0, len(X)):
        index = randint(0, len(A) - 1) #randomly selected index for the training point
        y_pred = model(A[index]) #networks prediction for the selected training point
        x = A[index] #the selected training point
        A = torch.cat([A[:index], A[index+1:]]) #we do sampling without replacement to train our network with the training points for each epoch
        #so update A by removing the selected training point

        deriv_net_x = torch.autograd.functional.jacobian(model, x)[0][0] #derivative of network with respect to x_1
        deriv_net_y = torch.autograd.functional.jacobian(model, x)[0][1] #derivative of network with respect to x_2
        second_deriv_net_x = torch.autograd.functional.hessian(model, x)[0][0] #second derivative of network with respect to x_1
        second_deriv_net_y = torch.autograd.functional.hessian(model, x)[1][1] #second derivative of network with respect to x_2

        #our current training point
        x1 = x[0] 
        x2 = x[1]

        #second derivatives of our solution u_t with respect to x_1 and x_2 as explained in the report on LinkedIn (equations are derived there)
        second_deriv_sol_x = 2*exp**(-x1)*(x2 - 1) - x2*(2*exp**(-x1) - exp**(-x1)*(x1 + 1)) - x1*exp**(-x1)*(x2 - 1) + 2*x2*y_pred*(x2 - 1) + 2*x1*x2*deriv_net_x*(x2 - 1) + 2*x2*deriv_net_x*(x1 - 1)*(x2 - 1) + x1*x2*(x1 - 1)*(x2 - 1)*second_deriv_net_x
        second_deriv_sol_y = 6*x1*x2*exp**(-1) - 6*x2*(x1 - 1) + 2*x1*y_pred*(x1 - 1) + 2*x1*x2*deriv_net_y*(x1 - 1) + 2*x1*deriv_net_y*(x1 - 1)*(x2 - 1) + x1*x2*(x1 - 1)*(x2 - 1)*second_deriv_net_y
        
        f = math.e**(-x1)*(x1 - 2 + x2**3 + 6*x2) #other part of the loss function, shown in report on LinkedIn

        loss = (second_deriv_sol_x + second_deriv_sol_y - f)**2 #our loss function
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() #we do backpropagation and update the network parameters

with torch.no_grad():
    nn_outputs = model(x_0) #get network outputs for our testing points

with torch.no_grad():
    nn_outputs_train = model(X) #get network outputs for our training points

#get solution u_t using network outputs for testing points
func = []
for i in range(0,len(nn_outputs)):
    X_0=x_0[i]
    x = X_0[0]
    y = X_0[1]
    func.append((1 - x)*y**3 + x*(1 + y**3)*math.e**(-1) + ((1 - y)*x)*(math.e**(-x) - math.e**(-1)) + y*((1 + x)*math.e**(-x) - (1 - x + 2*x*math.e**(-1))) + (x*(1-x)*y*(1-y))*nn_outputs[i])
func = torch.tensor(func, dtype=torch.float32)
func = func.numpy()

#get solution u_t using network outputs for training points
func_train = []
for i in range(0,len(nn_outputs_train)):
    X_0=X[i]
    x = X_0[0]
    y = X_0[1]
    func_train.append((1 - x)*y**3 + x*(1 + y**3)*math.e**(-1) + ((1 - y)*x)*(math.e**(-x) - math.e**(-1)) + y*((1 + x)*math.e**(-x) - (1 - x + 2*x*math.e**(-1))) + (x*(1-x)*y*(1-y))*nn_outputs_train[i])
func_train = torch.tensor(func_train, dtype=torch.float32)
func_train = func_train.numpy()

#split training and testing data into x and y coordinates for ease of use
x_0_test = x_0.numpy()
x_test = []
y_test = []
for i in range(0,len(x_0)):
    x_test.append(x_0_test[i][0])
    y_test.append(x_0_test[i][1])

X_train = X.numpy()
x_train = []
y_train = []
for i in range(0,len(X)):
    x_train.append(X_train[i][0])
    y_train.append(X_train[i][1])

#analytic solution of function u for testing points
z_test = []
for i in range(0, len(x_0)):
    z_test.append([analytic(x_test[i], y_test[i])])
z_test = torch.tensor(z_test, dtype=torch.float32)
z_test.numpy()

#analytic solution of function u for training points
z_train = []
for i in range(0, len(X)):
    z_train.append([analytic(x_train[i], y_train[i])])
z_train = torch.tensor(z_train, dtype=torch.float32)
z_train.numpy()

#calculate the loss mse and the test and train infinity norms and l2 norms 
norm_test = 0
z_test_norm = 0
inf_test = 0
for i in range(0, len(func)):
    norm_test += (func[i]- z_test[i])**2
    z_test_norm += (z_test[i])**2
    if abs(func[i]- z_test[i]) > inf_test:
        inf_test = abs(func[i]- z_test[i])
norm_test = math.sqrt(norm_test)
z_test_norm = math.sqrt(z_test_norm)

norm_train = 0
z_train_norm = 0
inf_train = 0
for i in range(0, len(func_train)):
    norm_train += (func_train[i]- z_train[i])**2
    z_train_norm += (z_train[i])**2
    if abs(func_train[i]- z_train[i]) > inf_train:
        inf_train = abs(func_train[i]- z_train[i])
norm_train = math.sqrt(norm_train)
z_train_norm = math.sqrt(z_train_norm)

rel_test = norm_test / z_test_norm
rel_train = norm_train / z_train_norm
loss_mse = norm_train / len(X)

print("Loss MSE train = " + str(loss_mse)) #loss mse when training
print("rel train = " + str(rel_train)) #l2 error training data
print("inf train = " + str(inf_train)) #infinity norm training data
print("rel test = " + str(rel_test)) #l2 error testing data
print("inf test = " + str(inf_test)) #infinity norm testing data

#plot the analytic solution and our predicted solution, u_t, using the network

torch.Tensor.ndim = property(lambda self: len(self.shape))

xaxis = arange(0, 1.01, 0.01)
yaxis = arange(0, 1.01, 0.01)

x,y = meshgrid(xaxis, yaxis)

#u_t predicted function
figure = pyplot.figure()
axis1 = figure.add_subplot(131,projection='3d')
axis1.set_title('u_t')
axis1.scatter(x_test,y_test,func,alpha=0.3)

#analytic function
z0 = analytic(x,y)
axis2 = figure.add_subplot(132,projection='3d')
axis2.set_title('Analytic')
axis2.plot_surface(x,y,z0,alpha=0.2)

figure.show()
