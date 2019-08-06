#-*- coding:utf-8 -*-
"""
@file: BP_Vec.py
@version:v1.0
@software:PyCharm

@author: fenglong.zhao
@contact: fenglong.zhao@irootech.com
@time:2019-5-23
"""
import numpy as np
from functools import *

class FullConnectedLayer(object):
    '''
    全连接层实现类
    '''
    def __init__(self, input_size, output_size,activator):
        '''
        构造函数
        input_size: 本层输入向量的维度
        output_size: 本层输出向量的维度
        activator: 激活函数
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # 权重数组W
        self.W = np.random.uniform(-0.1, 0.1,(output_size, input_size))
        # 偏置项b
        self.b = np.zeros((output_size, 1))
        # 输出向量
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        '''
        前向计算
        input_array: 输入向量，维度必须等于input_size
        '''
        self.input = input_array
        self.output = self.activator.forward(
            np.dot(self.W, input_array) + self.b)

    def backward(self, delta_array):
        '''
        反向计算W和b的梯度
        delta_array: 从上一层传递过来的误差项
        '''
        self.delta = self.activator.backward(self.input) * np.dot(
            self.W.T, delta_array)
        self.W_grad = np.dot(delta_array, self.input.T)
        self.b_grad = delta_array

    def update(self, learning_rate):
        '''
        使用梯度下降算法更新权重
        '''
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad

    def dump(self):
        print('W: %s\nb:%s' % (self.W, self.b))

class SigmoidActivator(object):
    '''
    Sigmoid激活函数类
    '''
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1 - output)

class Network(object):
    '''
    神经网络类
    '''
    def __init__(self, layers,eb=0.1,eta=0.5,mc=0.3,maxiter=10,iterator=10):
        '''
        构造函数
        '''
        self.eb = eb
        self.eta = eta
        self.mc = mc
        self.maxiter = maxiter
        self.errlist = []
        self.iterator = iterator
        self.layers = []

        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(
                    layers[i], layers[i+1],
                    SigmoidActivator()
                )
            )

    def train(self, labels, data_set,epoch):
        '''
        训练函数
        labels: 样本标签
        data_set: 输入样本
        epoch: 训练轮数
        '''
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d],data_set[d])

    def train_one_sample(self, label, sample):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(self.eta)

    def predict(self, sample):
        '''
        使用神经网络实现预测
        sample: 输入样本
        '''
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def calc_gradient(self, label):
        delta = self.layers[-1].activator.backward(
            self.layers[-1].output
        ) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)

    def dump(self):
        for layer in self.layers:
            layer.dump()

    def loss(self, label, output):
        return 0.5 * ((label - output) * (label - output)).sum()

    def gradient_check(self, sample_feature, sample_label):
        '''
        梯度检查
        network: 神经网络对象
        sample_feature: 样本的特征
        sample_label: 样本的标签
        '''

        # 获取网络在当前样本下每个连接的梯度
        self.predict(sample_feature)
        self.calc_gradient(sample_label)

        # 检查梯度
        epsilon = 10e-4
        for fc in self.layers:
            for i in range(fc.W.shape[0]):
                for j in range(fc.W.shape[1]):
                    fc.W[i, j] += epsilon
                    output = self.predict(sample_feature)
                    err1 = self.loss(sample_label, output)
                    fc.W[i, j] -= 2 * epsilon
                    output = self.predict(sample_feature)
                    err2 = self.loss(sample_label, output)
                    expect_grad = (err2 - err1) / (2 * epsilon)
                    fc.W[i, j] += epsilon
                    print('weights(%d,%d): expected - actural %.4e - %.4e' % (
                        i, j, expect_grad, fc.W_grad[i, j]))

class Normalizer(object):
    def __init__(self):
        self.mask = [0,1,2,3,4,5,6,7,8,9]

    def norm(self, number):
        data = list(map(lambda m: 0.9 if number == m else 0.1, self.mask))
        return np.array(data).reshape(10, 1)

    def denorm(self, vec):
        binary = list(map(lambda i: 1 if i > 0.5 else 0, vec[:,0]))
        for i in range(len(self.mask)):
            binary[i] = binary[i] * self.mask[i]
        return reduce(lambda x,y: x + y, binary)

class Loader(object):
    '''
    数据加载器基类
    '''
    def __init__(self, path, count):
        '''
        path: 数据文件路径
        count: 文件中的样本个数
        '''
        self.path = path
        self.count = count

    def get_file_content(self):
        '''
        读取文件内容
        '''
        f = open(self.path, 'rb')
        content = f.read()
        f.close()
        return content

    def transpose(self,args):
        return list(map(
            lambda line: np.array(line).reshape(len(line), 1)
            , args))

class FeatureLoader(Loader):
    '''
    特征数据加载器
    '''
    def get_feature(self, content, index):
        '''
        内部函数，从文件中获取图像
        '''
        start = index * 28 * 28 + 16
        feature = []
        for i in range(28):
            feature.append([])
            for j in range(28):
                feature[i].append(content[start + i * 28 + j])
        return feature

    def get_one_sample(self, picture):
        '''
        内部函数，将图像转化为样本的输入向量
        '''
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample

    def load(self):
        '''
        加载数据文件，获得全部样本的输入向量
        '''
        content = self.get_file_content()
        data_set = []
        for index in range(self.count):
            data_set.append(
                self.get_one_sample(
                    self.get_feature(content, index)))
        return self.transpose(data_set)

class LabelLoader(Loader):
    '''
    标签数据加载器
    '''
    def load(self):
        '''
        加载数据文件，获得全部样本的标签向量
        '''
        normalizer = Normalizer()
        content = self.get_file_content()
        labels = []
        for index in range(self.count):
            labels.append(normalizer.norm(content[index + 8]))
        return labels

def get_training_data_set():
    '''
    获得训练数据集
    '''
    feature_loader = FeatureLoader('train-images.idx3-ubyte', 60000)
    label_loader = LabelLoader('train-labels.idx1-ubyte', 60000)
    return feature_loader.load(), label_loader.load()

def get_test_data_set():
    '''
    获得测试数据集
    '''
    feature_loader = FeatureLoader('t10k-images.idx3-ubyte', 10000)
    label_loader = LabelLoader('t10k-labels.idx1-ubyte', 10000)
    return feature_loader.load(), label_loader.load()

def correct_ratio(network):
    normalizer = Normalizer()
    correct = 0.0
    for i in range(10):
        if normalizer.denorm(network.predict(normalizer.norm(i))) == i:
            correct += 1.0
    print('correct_ratio: %.2f%%' % (correct / 10))

def gradient_check():
    '''
    梯度检查
    '''
    data_set, labels  = get_test_data_set()
    net = Network([784, 90, 90, 10])
    net.gradient_check(data_set[0], labels[0])
    return net

def train():
    data_set, labels  = get_training_data_set()
    net = Network([784, 90, 90, 10])
    mini_batch = 1
    epoch = 3
    for i in range(epoch):
        net.train(labels, data_set,mini_batch)
        print('after epoch %d loss: %f' % (
            (i + 1),
            net.loss(labels[-1], net.predict(data_set[-1]))
        ))

    # correct_ratio(net)
    for i in net.layers:
        print(i.W)

if __name__ == '__main__':
    # gradient_check()
    train()
