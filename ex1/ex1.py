# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:29:39 2019

@author: 15366
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def CostFunction(X,y,theta):
    inner = np.power((np.dot(X , theta.T) - y),2)
    return np.sum(inner) / len(X) / 2

def gradientDescent(X,y,theta,alpha,epoch): #学习效率，训练次数
    temp = np.zeros(theta.shape)
   # num = int(theta.flatten().shape[1])#特征值的数量
    cost = np.zeros(epoch)
    m = X.shape[0]
    for i in range(epoch):
        temp = theta - (alpha/m)*np.dot((np.dot(X,theta.T)-y).T,X)
        theta = temp
        cost[i] = CostFunction(X,y,theta)
    return cost,theta    
 
    
epoch = 1000
alpha = 0.01
path = 'ex1data1.csv'
#names 用来添加列名，header为标题
data = pd.read_csv(path,header = None,names = ['Population','Profits'])
data.insert(0,'Ones',1)
col = data.shape[1]
X = data.iloc[:,0:col-1]
y = data.iloc[:,col-1:col]
theta = np.array([0,0]).reshape(1,2)

cost,ftheta = gradientDescent(X,y,theta,alpha,epoch)
print(CostFunction(X,y,ftheta))
x = np.linspace(data['Population'].min(),data['Population'].max(),100)
f = ftheta[0,0] + ftheta[0,1]*x
fig,ax= plt.subplots(figsize=(6,4))
ax.plot(x,f,'r')
ax.scatter(data['Population'],data['Profits'],label = 'training data')
ax.set_xlabel('Population')
ax.set_ylabel('Profit')

plt.show()