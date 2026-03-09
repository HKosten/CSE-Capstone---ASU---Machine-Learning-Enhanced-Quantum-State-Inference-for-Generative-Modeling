#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
import matplotlib_inline.backend_inline

matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


# In[2]:


# Loads the rolls.csv dataset
data = pd.read_csv('rolls.csv')
dice_rolls = data["roll"].values


# In[3]:


# Loads input and output features
lag = 2
X = []
y = []

for i in range(len(dice_rolls) - lag):
    X.append(dice_rolls[i:i+lag])
    y.append(dice_rolls[i+lag])

# In[4]:


# Splits the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)


# In[5]:


# Initializes and trains a classifier model on the training set
# This cell takes a long time to run.
model = MLPClassifier(
    random_state=1,
    max_iter=5000,
    hidden_layer_sizes=[10]
)
model.fit(X_train, y_train)


# In[6]:


# Predicts probability distribution of next roll given last rolls
last_rolls = [[2, 5]] 
probabilities = model.predict_proba(last_rolls)
print("Prob dist. for next roll:", probabilities)


# In[7]:


# Plots the bar chart for the training sets
plt.bar(range(1,7), probs[0])
plt.xlabel("Next dice roll")
plt.ylabel("Predicted probability")
plt.show()

# In[8]:


# Compare the final loss between train and test sets
pred_probs = model.predict_proba(X_test)
print("Log loss:", log_loss(y_test, pred_probs))

# In[9]:


# Obtains the final weights and biases


