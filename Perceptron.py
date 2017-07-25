#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 23:43:49 2017

@author: zony
"""
'''
实现感知器
学习网站:https://zybuluo.com/hanbingtao/note/433855
原始程序就python 2.x 编写
这里程序基于python 3.x 编写
'''
from functools import reduce
# python 3.x 中reduce函数已经移出了全局命名空间
# 定义感知器类
class Perceptron(object):
    def __init__(self,input_num,activator):
        '''
        初始化感知器，设置输入参数个数和激活函数
        激活函数的类型为double -> double
        '''
        self.activator = activator
        #权重向量初始化为零
        self.weights = [0.0 for _ in range(input_num)]
        #偏置项初始化为零
        self.bias = 0.0

    def __str__(self):
        '''
        打印学习到的权重和偏置项
        '''
        return 'weights \t:%s\n bias\t:%f\n' % (self.weights,self.bias)

    def predict(self,input_vec):
        '''
        输入向量:param input_vec:
        输出感知器结果:return:
        '''
        # 把input_vec[x1,x2,x3...]和weights[w1,w2,w3...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用map函数计算[x1*w1,x2*w2,x3*w3,...]
        # 最后利用reduce求和
        # 与原始程序有不同,python 3.x中的map函数不返回列表list,需要转化
        return self.activator(
            reduce(lambda a,b:a+b,
                   map(lambda wx:wx[0]*wx[1],
                       zip(self.weights,input_vec)))+self.bias)
    def train(self,input_vecs,labels,iteration,rate):
        '''
        训练数据:param input_vecs:
        训练数据的标签:param labels:
        训练次数:param iteration:
        学习率:param rate:
        :return:
        '''
        for i in range(iteration):
            self._one_iteration(input_vecs,labels,rate)

    def _one_iteration(self,input_vecs,labels,rate):
        '''
        一次迭代把所有的数据过一遍
        训练数据:param input_vecs:
        训练数据标签:param labels:
        学习率:param rate:
        :return:
        '''
        # 把输入和输出打包在一起，成为样本的列表[(input_vec,label),...]
        # 每个训练样本是(input_vec,label)
        samples = zip(input_vecs,labels)
        # 对于每个样本，按照感知器规则更新权重
        for (input_vec,label) in samples:
            # 计算感知器在当前权重下的输出
            output = self.predict(input_vec)
            # 更新权重
            self._update_weights(input_vec,output,label,rate)

    def _update_weights(self,input_vec,output,label,rate):
        '''
        按照感知器规则更新权重
        训练数据:param input_vec:
        训练数据标签:param label:
        学习率:param rate:
        :return:
        '''
        delta = label - output
        self.weights = list(map(lambda wx:wx[1]+rate*delta*wx[0],
                           zip(input_vec,self.weights)))
        # 更新bias
        self.bias = self.bias + rate * delta
def f(x):
    '''
    定义激活函数
    :param x:
    :return:
    '''
    return 1 if x > 0 else 0

def get_training_dataset():
    '''
    基于and真值表构建训练数据
    :return:
    '''
    #构建训练数据
    #输入向量列表
    input_vecs = [[1,1],[1,0],[0,1],[0,0]]
    #期望的输出列表，要与输入向量一一对应
    labels = [1,0,0,0]
    return input_vecs,labels
def train_and_perceptron():
    '''
    使用and真值表训练感知器
    :return:
    '''
    p = Perceptron(2,f)
    input_vecs,labels = get_training_dataset()
    p.train(input_vecs,labels,10,0.1)
    return p

if __name__ == '__main__':
    # 训练and 感知器
    and_perceptron = train_and_perceptron()
    # 打印训练得到的权重
    print(and_perceptron)
    # 测试
    print('1 and 1 = %d' % and_perceptron.predict([1, 1]))
    print('0 and 0 = %d' % and_perceptron.predict([0, 0]))
    print('1 and 0 = %d' % and_perceptron.predict([1, 0]))
    print('0 and 1 = %d' % and_perceptron.predict([0, 1]))
