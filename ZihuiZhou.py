#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt


# In[10]:


rating = pd.read_csv("ml-20m/ratings.csv")
rating.head()


# In[13]:


rating.columns


# In[21]:


# Neighborhood-based Collaborative Filtering

# Data preprocessing
rating['userId'] = rating['userId'].fillna(0)
rating['movieId'] = rating['movieId'].fillna(0)
rating['rating'] = rating['rating'].fillna(rating['rating'].mean())

# Start developing using a small dataset < 10,000 users / < 1,000 items
sample = rating.sample(frac = 0.01)
print(sample.info())

# Split the data into testing and training sets
train,test = train_test_split(sample, test_size = 0.2)


# In[27]:


# Use Pearson Correlation Coefficient to calculate the item similarity Matrix
train_matrix = train.as_matrix(columns = ['userId', 'movieId', 'rating'])
test_matrix = test.as_matrix(columns = ['userId', 'movieId', 'rating'])
item = 1 - pairwise_distances(train_matrix.T, metric = 'correlation')
item[np.isnan(item)] = 0


# In[32]:


# Function to predict
def predict(rating, similarity):
    predict = rating.dot(similarity)/np.array([np.abs(similarity).sum(axis=1)])
    return predict

# Evaluation using RMSE
def RMSE(predict, actual):
    predict = predict[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(predict,actual))

# Predict ratings of items
item_predict = predict(train_matrix, item)

# Calculate RMSE on the dataset
print('Item-based Collaborative Filtering RMSE on train data is:', RMSE(item_predict, train_matrix))


# In[ ]:




