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

#create training points in cartesian coordinates
X = []
for i in range(1,10):
    i=-1+i*0.2
    for j in range(1,5):
        j = j*0.2
        X.append([i,j])

for i in range(1,5):
    i = i*0.2
    for j in range(1,6):
        j = -1+j*0.2
        X.append([i,j])

#create testing points in cartesian coordinates
x_0 = []
for i in range(0,101):
    i=-1+i*0.02
    for j in range(0,51):
        j = j*0.02
        x_0.append([i,j])

for i in range(0,51):
    i = i*0.02
    for j in range(0,50):
        j = -1+j*0.02
        x_0.append([i,j])

#convert training points to polar coordinates
pi = math.pi
X_polar = []
for i in range(0,len(X)):
    x = X[i][0]
    y = X[i][1]

    r = math.sqrt((x)**2 + (y)**2)

    if x == 0 and y > 0:
        theta = pi/2
    else:
        theta = np.arctan2(y, x)
        
    X_polar.append([r, theta])

#convert testing points to polar coordinates
x_testing_polar = []
for i in range(0,len(x_0)):
    x = x_0[i][0]
    y = x_0[i][1]

    r = math.sqrt((x)**2 + (y)**2)

    if x == 0 and y > 0:
        theta = pi/2
    elif x == 0 and y < 0:
        theta = -pi/2
    else:
        theta = np.arctan2(y, x)
        
    x_testing_polar.append([r, theta])

x_0 = torch.tensor(x_0, dtype=torch.float32)
X = torch.tensor(X, dtype=torch.float32)
x_testing_polar = torch.tensor(x_testing_polar, dtype=torch.float32)
X_polar = torch.tensor(X_polar, dtype=torch.float32)

#our network architecture
model = nn.Sequential(
nn.Linear(2, 10),
nn.Sigmoid(),
nn.Linear(10, 1,bias=False),
)

optimizer = optim.Adam(model.parameters(), lr=0.001) #ADAM optimiser

n_epochs = 200 #number of epochs we train for
for epoch in range(n_epochs):
    A = torch.clone(X_polar) #used for sampling without replacement
    for i in range(0, len(X_polar)):
        index = randint(0, len(A) - 1)
        N = model(A[index])
        x = A[index]
        A = torch.cat([A[:index], A[index+1:]]) #remove used training point so can not be used again in this epoch

        r = x[0] #r value for current training point
        t = x[1] # theta value for current training point

        deriv_net_r = torch.autograd.functional.jacobian(model, x)[0][0] #derivative of network w.r.t. r
        deriv_net_t = torch.autograd.functional.jacobian(model, x)[0][1] #derivative of network w.r.t. theta
        second_deriv_net_r = torch.autograd.functional.hessian(model, x)[0][0] #second derivative of network w.r.t. r
        second_deriv_net_t = torch.autograd.functional.hessian(model, x)[1][1] #second derivative of network w.r.t. theta

        cos = np.cos(t) #cos of theta
        sin = np.sin(t) #sin of theta
            
        deriv_sol_r = (r*cos - 1)*(r*cos + 1)*(r*sin - 1)*(r*sin + 1)*N*(t - pi)*(t + pi/2) \
                      + r*cos*(r*cos - 1)*(r*sin - 1)*(r*sin + 1)*N*(t - pi)*(t + pi/2) + r*cos*(r*cos + 1)*(r*sin - 1)*(r*sin + 1)*N*(t - pi)*(t + pi/2) \
                      + r*sin*(r*cos - 1)*(r*cos + 1)*(r*sin - 1)*N*(t - pi)*(t + pi/2) + r*sin*(r*cos - 1)*(r*cos + 1)*(r*sin + 1)*N*(t - pi)*(t + pi/2) \
                      + r*(r*cos - 1)*(r*cos + 1)*deriv_net_r*(r*sin - 1)*(r*sin + 1)*(t - pi)*(t + pi/2)
        #derivative of u_t with respect to r
        
        second_deriv_sol_r = 2*r*((sin)**2)*(r*cos - 1)*(r*cos + 1)*N*(t - pi)*(t + pi/2) + 2*r*((cos)**2)*(r*sin - 1)*(r*sin + 1)*N*(t - pi)*(t + pi/2) \
                             + 2*cos*(r*cos - 1)*(r*sin - 1)*(r*sin + 1)*N*(t - pi)*(t + pi/2) \
                             + 2*cos*(r*cos + 1)*(r*sin - 1)*(r*sin + 1)*N*(t - pi)*(t + pi/2) \
                             + 2*sin*(r*cos - 1)*(r*cos + 1)*(r*sin - 1)*N*(t - pi)*(t + pi/2) \
                             + 2*sin*(r*cos - 1)*(r*cos + 1)*(r*sin + 1)*N*(t - pi)*(t + pi/2) \
                             + 2*(r*cos - 1)*(r*cos + 1)*deriv_net_r*(r*sin - 1)*(r*sin + 1)*(t - pi)*(t + pi/2) \
                             + r*(r*cos - 1)*(r*cos + 1)*(r*sin - 1)*(r*sin + 1)*(t - pi)*(t + pi/2)*second_deriv_net_r \
                             + 2*r*cos*sin*(r*cos - 1)*(r*sin - 1)*N*(t - pi)*(t + pi/2) + 2*r*cos*sin*(r*cos - 1)*(r*sin + 1)*N*(t - pi)*(t + pi/2) \
                             + 2*r*cos*sin*(r*cos + 1)*(r*sin - 1)*N*(t - pi)*(t + pi/2) + 2*r*cos*sin*(r*cos + 1)*(r*sin + 1)*N*(t - pi)*(t + pi/2) \
                             + 2*r*cos*(r*cos - 1)*deriv_net_r*(r*sin - 1)*(r*sin + 1)*(t - pi)*(t + pi/2) \
                             + 2*r*cos*(r*cos + 1)*deriv_net_r*(r*sin - 1)*(r*sin + 1)*(t - pi)*(t + pi/2) \
                             + 2*r*sin*(r*cos - 1)*(r*cos + 1)*deriv_net_r*(r*sin - 1)*(t - pi)*(t + pi/2) \
                             + 2*r*sin*(r*cos - 1)*(r*cos + 1)*deriv_net_r*(r*sin + 1)*(t - pi)*(t + pi/2)
        #second derivative of u_t w.r.t. r
        
        second_deriv_sol_theta = 2*r*(r*cos - 1)*(r*cos + 1)*(r*sin - 1)*(r*sin + 1)*N - 2*((r)**2)*sin*(r*cos - 1)*(r*sin - 1)*(r*sin + 1)*N*(t - pi) \
                                 - 2*((r)**2)*sin*(r*cos + 1)*(r*sin - 1)*(r*sin + 1)*N*(t - pi)  \
                                 - 2*((r)**2)*sin*(r*cos - 1)*(r*sin - 1)*(r*sin + 1)*N*(t + pi/2) \
                                 - 2*((r)**2)*sin*(r*cos + 1)*(r*sin - 1)*(r*sin + 1)*N*(t + pi/2) \
                                 + 2*r*(r*cos - 1)*(r*cos + 1)*deriv_net_t*(r*sin - 1)*(r*sin + 1)*(t - pi) \
                                 + 2*r*(r*cos - 1)*(r*cos + 1)*deriv_net_t*(r*sin - 1)*(r*sin + 1)*(t + pi/2) \
                                 + 2*((r)**3)*((cos)**2)*(r*cos - 1)*(r*cos + 1)*N*(t - pi)*(t + pi/2) \
                                 + 2*((r)**3)*((sin)**2)*(r*sin - 1)*(r*sin + 1)*N*(t - pi)*(t + pi/2) \
                                 + 2*((r)**2)*cos*(r*cos - 1)*(r*cos + 1)*(r*sin - 1)*N*(t - pi) \
                                 + 2*((r)**2)*cos*(r*cos - 1)*(r*cos + 1)*(r*sin + 1)*N*(t - pi) \
                                 + 2*((r)**2)*cos*(r*cos - 1)*(r*cos + 1)*(r*sin - 1)*N*(t + pi/2) \
                                 + 2*((r)**2)*cos*(r*cos - 1)*(r*cos + 1)*(r*sin + 1)*N*(t + pi/2) \
                                 + 2*((r)**2)*cos*(r*cos - 1)*(r*cos + 1)*deriv_net_t*(r*sin - 1)*(t - pi)*(t + pi/2) \
                                 + 2*((r)**2)*cos*(r*cos - 1)*(r*cos + 1)*deriv_net_t*(r*sin + 1)*(t - pi)*(t + pi/2) \
                                 - 2*((r)**3)*cos*sin*(r*cos - 1)*(r*sin - 1)*N*(t - pi)*(t + pi/2) \
                                 - 2*((r)**3)*cos*sin*(r*cos - 1)*(r*sin + 1)*N*(t - pi)*(t + pi/2) \
                                 - 2*((r)**3)*cos*sin*(r*cos + 1)*(r*sin - 1)*N*(t - pi)*(t + pi/2) \
                                 - 2*((r)**3)*cos*sin*(r*cos + 1)*(r*sin + 1)*N*(t - pi)*(t + pi/2) \
                                 - 2*((r)**2)*sin*(r*cos - 1)*deriv_net_t*(r*sin - 1)*(r*sin + 1)*(t - pi)*(t + pi/2) \
                                 - 2*((r)**2)*sin*(r*cos + 1)*deriv_net_t*(r*sin - 1)*(r*sin + 1)*(t - pi)*(t + pi/2) \
                                 - ((r)**2)*cos*(r*cos - 1)*(r*sin - 1)*(r*sin + 1)*N*(t - pi)*(t + pi/2) \
                                 - ((r)**2)*cos*(r*cos + 1)*(r*sin - 1)*(r*sin + 1)*N*(t - pi)*(t + pi/2) \
                                 - ((r)**2)*sin*(r*cos - 1)*(r*cos + 1)*(r*sin - 1)*N*(t - pi)*(t + pi/2) \
                                 - ((r)**2)*sin*(r*cos - 1)*(r*cos + 1)*(r*sin + 1)*N*(t - pi)*(t + pi/2) \
                                 + r*(r*cos - 1)*(r*cos + 1)*(r*sin - 1)*(r*sin + 1)*(t - pi)*(t + pi/2)*second_deriv_net_t
        #second derivative of u_t w.r.t. theta

        loss = (second_deriv_sol_r + (deriv_sol_r)/r + (second_deriv_sol_theta)/((r)**2) + 1)**2 #our loss function to be minimised
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() #update network parameters using backpropagation

with torch.no_grad():
    nn_outputs = model(x_testing_polar) #network outputs for testing polar coordinate points

func = []
for i in range(0,len(nn_outputs)):
    X_0=x_testing_polar[i]
    r = X_0[0]
    theta = X_0[1]
    cos = np.cos(theta)
    sin = np.sin(theta)
    func.append(nn_outputs[i]*(theta-pi)*(theta+(pi/2))*r*(r*cos - 1)*(r*sin - 1)*(r*cos + 1)*(r*sin + 1)) #u_t values for testing polar coordinate points
    #(our estamated value of the function u_t using the network outputs)
func = torch.tensor(func, dtype=torch.float32)

#preparing data for plotting

func = func.numpy()
        
torch.Tensor.ndim = property(lambda self: len(self.shape))

x_0_test = x_0.numpy()
x_test = []
y_test = []
for i in range(0,len(x_0)):
    x_test.append(x_0_test[i][0])
    y_test.append(x_0_test[i][1])
        
figure = pyplot.figure()
axis = figure.add_subplot(projection='3d')
axis.scatter(x_test,y_test,func,alpha=0.3) #plot u_t testing point values at x_1, x_2 coordinates

plt.show()

#save model to file if we are happy with it, so can be loaded later on
#torch.save(model, PATH)

