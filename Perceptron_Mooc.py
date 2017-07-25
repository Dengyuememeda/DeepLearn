#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 08:39:50 2017

@author: zony
"""
'''
慕课网课程《机器学习-实现简单神经网络》
感知器实现
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from matplotlib.colors import ListedColormap
 

class Perceptron(object):
    '''
    lr:learning rate，学习速率，代表权系数和偏置项每次变化的大小
    n_iter:迭代次数
    w_:神经分叉权重向量
    errors_:用于记录神经元判断出错次数
    '''
    def __init__(self,lr = 0.01,n_iter = 10):
        '''
        这里不初始化权重向量的原因是，初始化权重向量需要知道训练数据的特征维数
        '''
        self.lr = lr
        self.n_iter = n_iter
        pass
    def fit(self,X,y):
        '''
        X:输入训练数据，shape(n_samples,n_features)
        y:训练数据的标签，对应分类
        '''
        self.w_ = np.zeros(1+X.shape[1])# 初始化权重向量，包含率偏置项对应的权重系数，1
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.lr * (target - self.predict(xi))
                self.w_[1:] = self.w_[1:] + xi * update
                self.w_[0] = update
                errors += int(update !=0.0) 
                self.errors_.append(errors)
                pass
            pass
        pass
    
    def predict(self,X):
        return np.where(self.net_input(X) >= 0.0, 1,-1)
        pass
    def net_input(self,X):
        '''
        z = w0*1 + w1*x1 + w2*x2 +...
        '''
        return np.dot(X,self.w_[1:])+self.w_[0]
        pass
    pass
def plot_decision_regions(X,y,classifier,resolution=0.02):
    marker = ('s','x','o','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min,x1_max = X[:,0].min()-1,X[:,0].max()
    x2_min,x2_max = X[:,1].min()-1,X[:,1].max()
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
                          np.arange(x2_min,x2_max,resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    
    print(xx1.ravel())
    print(xx2.ravel())
    print(Z)
    
    z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,z,alpha=0.4,cmap = cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    
    for idx,cl in enumerate(np.unique(y)):
        '''
        np.unique(y)：返回y中不重复的元素，
        利用enumerate()可以同时获得索引idx和对应的值cl
        X[y==cl,0],获得对应标签为cl的X的数据
        '''
        plt.scatter(x=X[y==cl,0],y = X[y==cl,1],alpha = 0.8, c=cmap(idx),
                    marker=marker[idx],label=cl)
        
        pass
    pass



if __name__ == '__main__':
    iris = datasets.load_iris()
    print(iris.data)
    print(iris.target)
    
    print(iris.data[:100,[0,2]])
    print(iris.target[:100])
    
    X = iris.data[:100,[0,2]]
    y = iris.target[:100]
    
    plt.scatter(X[:50,0],X[:50,1],color='red',marker='o',label='setosa')
    plt.scatter(X[50:100,0],X[50:100,1],color='blue',marker='x',label='versicolor')
    plt.xlabel('huabanchangdu')
    plt.ylabel('huajingchangdu')
    plt.show()
    
    p = Perceptron(0.01,10)
    p.fit(X,y)
    
    print(p.w_)

    plot_decision_regions(X,y,p,resolution=0.02)
    plt.legend(loc='upper left')
    plt.show()
    
