# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:38:24 2019

@author: 15366
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def CostFunction(X,y,theta):
    h = np.power(X * theta.T - y,2)
    l = X.shape[0]
    return np.sum(h)/2/l

def gradientDescent(X,y,theta,alpha,eponch):
    tem = np.matrix(np.zeros(data.shape[1]-1))
    l = X.shape[0]
    cost = np.zeros(eponch)
    for i in range(eponch):
        tem = theta - (alpha / l) * (X* theta.T - y).T * X
        theta = tem
        cost[i] = CostFunction(X,y,theta)
        
    return theta,cost

def normalEqn(X, y):#标准化
    theta = np.linalg.inv(X.T@X)@X.T@y#X.T@X等价于X.T.dot(X)
    return theta


alpha = 0.1
eponch = 1000
path = 'ex1data2.csv'
data = pd.read_csv(path,header = None,names = ['Size','Nums','Prince'])
data = (data - data.mean()) / data.std()
data.insert(0,'Ones',1)
col = data.shape[1]
theta = np.matrix(np.zeros(data.shape[1]-1))
X = data.iloc[:,0:col-1]
y = data.iloc[:,col-1:col]
X = np.matrix(X.values)
y = np.matrix(y.values)
ftheta,cost = gradientDescent(X,y,theta,alpha,eponch)
ntheta = normalEqn(X, y)
#plt.scatter(data['Nums'],data['Prince'])
#plt.show()