#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib_inline.backend_inline

matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


# In[2]:


# Loads the homes.csv dataset
homes = pd.read_csv('homes.csv')
homes


# In[3]:


# Loads input and output features
X = homes[['Bed', 'Floor']]
y = homes[['Price']]


# In[4]:


# Splits the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(y), random_state=123)


# In[5]:


# Initializes and trains a multilayer perceptron regressor model on the training set
# This cell takes a long time to run.
mlpReg_train = MLPRegressor(
    random_state=1, max_iter=500000, hidden_layer_sizes=[1]
).fit(X_train, np.ravel(y_train))


# In[6]:


# Predicts the price of a 5 bedroom house with 2,896 sq ft
mlpReg_train.predict([[5, 2.896]])


# In[7]:


# Plots the loss curves for the training sets
f, ax = plt.subplots(1, 1)
sns.lineplot(
    x=range(len(mlpReg_train.loss_curve_)), y=mlpReg_train.loss_curve_, label='Training'
)
ax.set_xlabel('Epochs', fontsize=14);
ax.set_ylabel('Loss', fontsize=14);


# In[8]:


# Compare the final loss between train and test sets
print(mlpReg_train.loss_)
print(
    mean_squared_error(y_test, mlpReg_train.predict(X_test)) / 2
)  # division by 2 to get squared error to match squared error.


# In[9]:


# Obtains the final weights and biases
print(mlpReg_train.coefs_)
print(mlpReg_train.intercepts_)

