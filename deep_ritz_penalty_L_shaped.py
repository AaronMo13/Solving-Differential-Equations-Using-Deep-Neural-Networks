import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from numpy.random import random
from numpy.random import randint
from numpy import arange
from numpy import meshgrid
from matplotlib import pyplot

#generate set of internal training points
internal_training_points = []

for i in range(0,1000):
    x = -1 + random()*2
    y = 0 + random()
    if x != -1 and y != 0:
        internal_training_points.append([x,y])

for i in range(0,500):
    x =  random()
    y = -1 + random()
    if x != 0 and y != -1:
        internal_training_points.append([x,y])

internal_training_points = torch.tensor(internal_training_points, dtype=torch.float32)

#generate set of boundary training points
boundary_training_points = []

for i in range(0,50):
    x = -1
    y = random()
    boundary_training_points.append([x,y])

for i in range(0,100):
    x = 1
    y = -1 + random()*2
    boundary_training_points.append([x,y])

for i in range(0,50):
    x = random()
    y = -1
    boundary_training_points.append([x,y])

for i in range(0,100):
    x = -1 + random()*2
    y = 1
    boundary_training_points.append([x,y])

for i in range(0,50):
    x = 0
    y = -1 + random()
    boundary_training_points.append([x,y])

for i in range(0,50):
    x = -1 + random()
    y = 0
    boundary_training_points.append([x,y])

boundary_training_points = torch.tensor(boundary_training_points, dtype=torch.float32)

#generate set of testing points
x_0=[]
for i in range(0,101):
    i= -1 + i*0.02
    for j in range(0,51):
        j= j*0.02
        x_0.append([i,j])

for i in range(0,51):
    i = 0 + i*0.02
    for j in range(0,51):
        j = -1 + j*0.02
        x_0.append([i,j])
        
x_0 = torch.tensor(x_0, dtype=torch.float32)

#our network architecture
model = nn.Sequential(
nn.Linear(2, 10),
nn.Sigmoid(),
nn.Linear(10,10),
nn.Sigmoid(),
nn.Linear(10,10),
nn.Sigmoid(),
nn.Linear(10,10),
nn.ReLU(),
nn.Linear(10, 1,bias=False),
)

optimizer = optim.Adam(model.parameters(), lr=0.001) #using ADAM optimiser
batch_size_internal = 150 #batch size for internal training points
batch_size_boundary = 30 #batch size for boundary training points

beta = 100 #our beta penalty term value

a = -1 + random()*2 #randomise initial value of a
b = -1 + random()*2 #randomise initial value of b

iterations = 1000 #we use 1000 iterations
for j in range(iterations):
    g_internal = 0
    g_boundary = 0
    g_a_internal=0
    g_a_boundary = 0
    g_b_boundary = 0
    for i in range(0,batch_size_internal): #get g_I value for internal batch number of points
        rand_index = randint(0,1500) #select a random training point index from the internal points
        x = internal_training_points[rand_index] #random internal training point
        y_pred = model(x) #the neural network output of our random internal training point
        
        deriv_net_x = torch.autograd.functional.jacobian(model, x)[0][0] #derivative of network with respect to x_1
        deriv_net_y = torch.autograd.functional.jacobian(model, x)[0][1] #derivative of network with respect to x_2

        deriv_sol_x = a*deriv_net_x #derivative of solution u_e with respect to x_1
        deriv_sol_y = a*deriv_net_y #derivative of solution u_e with respect to x_2

        g_internal += ((1/2)*((a)**2))*((deriv_sol_x)**2 + (deriv_sol_y)**2) - (a*y_pred) - b #increase g_I value by adding g_I value for our randomly selected training point

        with torch.no_grad():
            g_a_internal += a*((deriv_net_x)**2 + (deriv_net_y)**2) - y_pred #increase partial derivative total with respect to a value, by adding partial derivative with respect to a value for current training point
            
    for i in range(0,batch_size_boundary): #get g_B value for boundary batch number of points
        rand_index = randint(0,400) #select a random training point index from the boundary points
        x = boundary_training_points[rand_index] #random boundary training point
        y_pred = model(x) #the neural network output of our random boundary training point

        g_boundary += (a)**2*(y_pred)**2 + 2*a*b*y_pred + (b)**2 #increase g_B value by adding g_B value for our randomly selected training point

        with torch.no_grad():
            g_a_boundary += 2*a*(y_pred)**2 + 2*b*y_pred #increase partial derivative with respect to a value
            g_b_boundary += 2*a*y_pred + 2*b #increase partial derivative with respect to b value

    loss = (g_internal/batch_size_internal + beta*(g_boundary/batch_size_boundary)) #calculate total g value (our loss function we want to minimise)
    optimizer.zero_grad() 
    loss.backward() #backpropagation (work out gradient of loss function with respect to network parameters)
    optimizer.step() #update our network parameters
    a -= 0.001*(g_a_internal/batch_size_internal + (beta*(g_a_boundary))/batch_size_boundary) #update a
    b -= 0.001*(-1 + (beta*(g_b_boundary))/batch_size_boundary) #update b

with torch.no_grad():
    nn_outputs = model(x_0) #outputs of network for our testing points

func = []
for i in range(0,len(nn_outputs)):
    X_0=x_0[i]
    x = X_0[0]
    y = X_0[1]
    func.append(a*nn_outputs[i] + b) #calculate u_e values for our testing points
func = torch.tensor(func, dtype=torch.float32)

#preparing data for plotting

func = func.numpy()

torch.Tensor.ndim = property(lambda self: len(self.shape))

xaxis = arange(0, 1.01, 0.01)
yaxis = arange(0, 1.01, 0.01)

x,y = meshgrid(xaxis, yaxis)

x_0_test = x_0.numpy()
x_test = []
y_test = []
for i in range(0,len(x_0)):
    x_test.append(x_0_test[i][0])
    y_test.append(x_0_test[i][1])
        
figure = pyplot.figure()
axis = figure.add_subplot(projection='3d')
axis.scatter(x_test,y_test,func,alpha=0.3) #plot our u_e values for testing points

figure.show()

#save model to file if we are happy with it, so can be loaded later on
#torch.save(model, PATH)

        
