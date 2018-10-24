import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tools.eval_measures import rmse
import matplotlib.pylab as plt
from sklearn.preprocessing import scale

# Read Boston Housing Data
data = pd.read_csv('Housing.csv')

# Data Summary
data.describe()

# remove the chas colmn
# more detail here - https://www.kaggle.com/c/boston-housing#description
data = data.drop(["chas"],axis=1)

# check for null vvalue
data.info()

x_lable = ["Per Capita Crime","Residential Land Over 25,000 Sq.ft",
           "Non-Retail Business Acer Ratio","Nitrogen Oxides Concentration (PPM)",
           "Average Number of Rooms Per House","Units ratio before 1940",
           "Distance (mean) to the Employment Center","Radial Highways Accessibility Index",
           "Property Tax Rate","Teacher-Student Ratio","Black People Ration",
           "Lower Status population %","Media Value of the Hosue"]
y_lable = "Number of Town"

# plt histogram for the data points
f = plt.figure(figsize=(15,15))
for index in range(len(data.columns)):
    location = int("45"+str(index+1))
    plt1 = f.add_subplot(4,5,index+1)
    plt1.hist(data[data.columns[index]],bins=20,density=True)
    plt1.set_xlabel(x_lable[index])
    plt1.grid(True)
    plt1.set_ylabel(y_lable)
f.show()

import seaborn as sns
plt.figure(figsize=(14,14))
correlations = data.corr()
plt.title("Correlation heatmap for housing features")
temp = sns.heatmap(correlations, cbar = True,  square = True, annot=True,
            fmt= '.2f',annot_kws={'size': 12}, cmap= 'coolwarm')
plt.show()
# See the Correlation between values
X_input = data.iloc[:,:12].values
y = data.iloc[:,-1].values

## Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_input,y,test_size = 0.20)
l_rate = 0.000001
iteration = 1000
### Pytorch
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
class LinearRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):

        super(LinearRegressionModel, self).__init__()
        # Calling Super Class's constructor
        self.linear = nn.Linear(input_dim, output_dim)
        # nn.linear is defined in nn.Module

    def forward(self, x):
        # Here the forward pass is simply a linear function

        out = self.linear(x)
        return out
def train(model,X_train,y_train,iteration,optimiser,criterion):
    '''
    Model: ML model to train data
    X_train and Y_train: Input Data
    iteration: number of epoch through data
    optimiser: to minimize the cost function

    return : only loss values per epoch

    '''
    loss_ = []
    for epoch in range(iteration):
        epoch +=1
        #increase the number of epochs by 1 every time
        inputs = Variable(torch.from_numpy(X_train))
        labels = Variable(torch.from_numpy(y_train)).reshape(-1,1)

        #clear grads as discussed in prev post
        optimiser.zero_grad()
        #forward to get predicted values
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()# back props
        optimiser.step()# update the parameters
        loss_.append(loss.item())

    return loss_
# First model using mse

model = LinearRegressionModel(X_train.shape[1],1).double()
optimiser = optim.SGD(model.parameters(), lr = l_rate) #Stochastic Gradient Descent
criterion = nn.MSELoss()# Mean Squared Loss

loss_mse_with_out_data_norm = train(model,X_train,y_train,iteration,optimiser
                                    ,criterion)

model = LinearRegressionModel(X_train.shape[1],1).double()
criterion = nn.L1Loss()# L1 Loss
optimiser = optim.SGD(model.parameters(), lr = l_rate) #Stochastic Gradient Descent

loss_l1_with_out_data_norm = train(model,X_train,y_train,iteration,optimiser
                                    ,criterion)
# Loss with normalization

X_input = scale(data.iloc[:,:12].values)
y = data.iloc[:,-1].values

model = LinearRegressionModel(X_train.shape[1],1).double()
optimiser = optim.SGD(model.parameters(), lr = l_rate) #Stochastic Gradient Descent
criterion = nn.MSELoss()# Mean Squared Loss

loss_mse_with_data_norm = train(model,X_train,y_train,iteration,optimiser
                                    ,criterion)

model = LinearRegressionModel(X_train.shape[1],1).double()
optimiser = optim.SGD(model.parameters(), lr = l_rate) #Stochastic Gradient Descent
criterion = nn.L1Loss()# Mean Squared Loss

loss_l1_with_data_norm = train(model,X_train,y_train,iteration,optimiser
                                    ,criterion)
from math import log10
def scale_down_loss_val(data):
    return list(map(log10,data))
#plt.plot(loss_, "c-", label="Precision")
#plt.plot(loss_1, "c-", label="Precision")
#plt.rcParams['axes.facecolor'] = '#d5d9e0'
plt.figure(figsize=(9, 9))
plt.title("Loss with L1 and MSE Error")
plt.plot(scale_down_loss_val(loss_mse_with_out_data_norm), "c-", label="Loss (No Data Norm) with MSE")
plt.plot(scale_down_loss_val(loss_l1_with_out_data_norm), "g-", label="Loss (No Data Norm) with L1")
plt.ylabel("Loss (Log Scale)")
plt.xlabel("Iteratino")
plt.legend(loc='best')
plt.grid()
plt.show()


plt.figure(figsize=(9, 9))
plt.title("Loss with L1 and MSE Error")
plt.plot(scale_down_loss_val(loss_mse_with_data_norm), "#395c93", label="Loss (With Data Norm) with MSE")
plt.plot(scale_down_loss_val(loss_l1_with_data_norm), "#247a4f", label="Loss (With Data Norm) with L1")
plt.ylabel("Loss (Log Scale)")
plt.xlabel("Iteratino")
plt.legend(loc='best')
plt.grid()
plt.show()
