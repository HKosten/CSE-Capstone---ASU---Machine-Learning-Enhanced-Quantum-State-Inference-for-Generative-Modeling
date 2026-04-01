'''
We will use a simple multi-layer perceptron network in this lab to learn a quadratic function X^2 = Y` where `x` is a single dimension input variable and Y is a single dimension target variable.  
Name this file mlp.py.

We will use pytorch in this in-class activity, you can use command 'pip install torch' to install torch on your local machine to develop your code or you can just use colab where torch is pre-installed.

The code needs to be implemented is labeled TO DO !!!!!, there is not much for you to implement but do learn about the entire workflow.

Name your file mlp.py when submitting to GradeScope.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

#Set seeds 
torch.manual_seed(42)
np.random.seed(42)

#Generate 1000 dice rolls
rolls = np.random.randint(1,7, 1000)

#Use last 2 rolls to predict next roll (autoregression)
#x = features, each element represents 2 previous rolls
#y = targets, each element represents the next roll after the 2 prev. rolls
X_data = []
y_data = []

for i in range(len(rolls)-2):
    X_data.append(rolls[i:i+2])
    y_data.append(rolls[i+2])

#Convert data lists to tensors
#CrossEntropyLoss expects class indices starting at 0, subtract 1 from y_data
X_data = torch.tensor(X_data, dtype=torch.float32)
X_data = X_data/6
y_data = torch.tensor(y_data, dtype=torch.long)
y_data = y_data - 1 

#Form MLP from 2 single layer perceptrons
#First layer - map 2 inputs (lag size) to 10 hidden neurons (intermediate dimension = 10)
#Second layer - map 10 hidden neurons to 6 ouputs (for dice faces)
#forward - how data passes thru network. Pass thru first layer, pass thru ReLU activation func., 
# pass thru second layer (logits)
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # Here you will need to define two single-layer perceptrons (with non-linearality) to form a two-layer perceptron network.
        # Input is 1 dimension, we will make the intermediate dimension to be 10, the final output dimension is also 1. Below is the network:
        # X (1 dimension) -> 10 dimensional internal layer -> Relu -> Y (1 dimension)
        # We use nn.Linear for a single-layer perceptron as in the previous ICE.
        # For ReLU implementation, we will use F.relu (https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html)
        self.fc1 =  nn.Linear(2,32)# TO DO !!!!!
        self.fc2 =  nn.Linear(32,6)# TO DO !!!!!

    def forward(self, x):
        out =  self.fc1(x)# TO DO !!!!! pass through self.fc1
        out =  F.relu(out)# TO DO !!!!! pass through F.relu
        out =  self.fc2(out)# TO DO !!!!! pass through self.fc2

        return out# TO DO !!!!!


net = Net()

#Use Adam instead of SGD
optimizer = optim.Adam(net.parameters(), lr=0.05) # TO DO !!!!!

# train 10 epochs
# use CrossEntropyLoss() for multi-class classification instead of MSE
for epoch in range(100): # TO DO !!!!!
    optimizer.zero_grad()

    output = net(X_data) # TO DO !!!!!

    loss = nn.CrossEntropyLoss()(output,y_data) # TO DO !!!!!

    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch+1} - loss: {loss.item():.4f}")

# let's test our model
# predict probability dist. for next roll if last rolls were [2, 5]
last_rolls = torch.tensor([[2,5]], dtype=torch.float32)
output = net(last_rolls)
probabilities = F.softmax(output, dim=1).detach().numpy()
print("Predicted probability distribution for next roll:", probabilities)
