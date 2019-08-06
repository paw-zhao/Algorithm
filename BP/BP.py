#-*- coding:utf-8 -*-
"""
@file: BP.py
@version:v1.0
@software:PyCharm

@author: fenglong.zhao
@contact: fenglong.zhao@irootech.com
@time:2019-5-20
"""
from functools import *
import math
import random
from datetime import datetime

class Node(object):
    '''
    节点类，负责记录和维护节点自身信息以及与这个节点相关的上下游连接，实现输出值和误差项的计算
    '''
    def __init__(self, layer_index, node_index):
        '''
        :param layer_index: 节点所属的层编号
        :param node_index: 节点的编号
        '''
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        '''
        设置节点的输出值。如果节点属于输入层会用到这个函数
        '''
        self.output = output

    def append_downstream_connection(self, conn):
        '''
        添加一个到下游节点的连接
        '''
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        '''
        添加一个到上游节点的连接
        '''
        self.upstream.append(conn)

    def calc_output(self):
        '''
        计算节点的输出
        '''
        def sigmoid(x):
            return 1.0 / (1.0 + math.exp(-x))
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream,0.0)
        try:
            self.output = sigmoid(output)
        except OverflowError:
            self.output = 0

    def calc_hidden_layer_delta(self):
        '''
        节点属于隐藏层时，计算delta
        '''
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calc_output_layer_delta(self, label):
        '''
        节点属于输出层时，计算delta
        '''
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        '''
        打印节点的信息
        '''
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str

class ConstNode(object):
    '''
    构造一个输出恒为1的节点(计算偏置项时需要)
    '''
    def __init__(self, layer_index, node_index):
        '''
        :param layer_index: 节点所属的层编号
        :param node_index: 节点编号
        '''
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1

    def append_downstream_connection(self, conn):
        '''
        添加一个到下游节点的连接
        '''
        self.downstream.append(conn)

    def calc_hidden_layer_delta(self):
        '''
        节点属于隐藏层时，计算delta
        '''
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self):
        '''
        打印节点的信息
        '''
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str

class Layer(object):
    '''
    负责初始化一层。此外，作为Node的集合对象，提供对Node集合的操作
    '''
    def __init__(self, layer_index, node_count):
        '''
        :param layer_index: 层编号
        :param node_count: 层所包含的节点个数
        '''
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count))

    def set_output(self, data):
        '''
        设置层的输出。当层是输入层时会用到
        '''
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        '''
        计算层的输出向量
        '''
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        '''
        打印层的信息
        '''
        for node in self.nodes:
            print(node)

class Connection(object):
    '''
    记录连接的权重，以及这个连接所关联的上下游节点
    '''
    def __init__(self, upstream_node, downstream_node):
        '''
        初始化连接，权重初始化为是一个很小的随机数
        upstream_node: 连接的上游节点
        downstream_node: 连接的下游节点
        '''
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1, 0.1)
        self.gradient = 0.0

    def calc_gradient(self):
        '''
        计算梯度
        '''
        self.gradient = self.downstream_node.delta * self.upstream_node.output

    def get_gradient(self):
        '''
        获取当前的梯度
        '''
        return self.gradient

    def update_weight(self, rate):
        '''
        根据梯度下降算法更新权重
        '''
        self.calc_gradient()
        self.weight += rate * self.gradient

    def __str__(self):
        '''
        打印连接信息
        '''
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight)

class Connections(object):
    '''
    提供Connection集合
    '''
    def __init__(self):
        self.connections = []
    def add_connection(self, connection):
        self.connections.append(connection)
    def dump(self):
        for conn in self.connections:
            print(conn)

class Network(object):
    '''
    构造Network对象，提供API
    '''
    def __init__(self, layers):
        '''
        初始化一个全连接神经网络
        layers: 列表，描述神经网络每层节点数
        '''
        self.connections = Connections()
        self.layers = []
        layer_count = len(layers)
        for i in range(layer_count):
            self.layers.append(Layer(i, layers[i]))
        for layer in range(layer_count - 1):
            connections = [Connection(upstream_node, downstream_node)
                           for upstream_node in self.layers[layer].nodes
                           for downstream_node in self.layers[layer + 1].nodes[:-1]]
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)

    def train(self, labels, data_set, rate, iteration):
        '''
        训练神经网络
        labels: 数组，训练样本标签。每个元素是一个样本的标签。
        data_set: 二维数组，训练样本特征。每个元素是一个样本的特征。
        '''
        for i in range(iteration):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        '''
        内部函数，用一个样本训练网络
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    def predict(self, sample):
        '''
        根据输入的样本预测输出值
        sample: 数组，样本的特征，也就是网络的输入向量
        '''
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return list(map(lambda node: node.output, self.layers[-1].nodes[:-1]))

    def calc_delta(self, label):
        '''
        内部函数，计算每个节点的delta
        '''
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self, rate):
        '''
        内部函数，更新每个连接权重
        '''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def calc_gradient(self):
        '''
        内部函数，计算每个连接的梯度
        '''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def get_gradient(self, label, sample):
        '''
        获得网络在一个样本下，每个连接上的梯度
        label: 样本标签
        sample: 样本输入
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def dump(self):
        '''
        打印网络信息
        '''
        for layer in self.layers:
            layer.dump()

def gradient_check(network, sample_feature, sample_label):
    '''
    梯度检查
    network: 神经网络对象
    sample_feature: 样本的特征
    sample_label: 样本的标签
    '''
    # 计算网络误差
    network_error = lambda vec1, vec2: \
            0.5 * reduce(lambda a, b: a + b,
                      map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                          zip(vec1, vec2)))
    # 获取网络在当前样本下每个连接的梯度
    network.get_gradient(sample_label,sample_feature)
    # 对每个权重做梯度检查
    for conn in network.connections.connections:
        # 获取指定连接的梯度
        actual_gradient = conn.get_gradient()
        # 增加一个很小的值，计算网络的误差
        epsilon = 0.0001
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature), sample_label)
        # 减去一个很小的值，计算网络的误差
        conn.weight -= 2 * epsilon # 刚才加过了一次，因此这里需要减去2倍
        error2 = network_error(network.predict(sample_feature), sample_label)
        # 根据式6计算期望的梯度值
        expected_gradient = (error2 - error1) / (2 * epsilon)
        # 打印
        print('expected gradient: \t%f\nactual gradient: \t%f' % (
            expected_gradient, actual_gradient))

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

class ImageLoader(Loader):
    '''
    图像数据加载器
    '''
    def get_picture(self, content, index):
        '''
        内部函数，从文件中获取图像
        '''
        start = index * 28 * 28 + 16
        picture = []
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(content[start + i * 28 + j])
        return picture

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
                    self.get_picture(content, index)))
        return data_set

class LabelLoader(Loader):
    '''
    标签数据加载器
    '''
    def load(self):
        '''
        加载数据文件，获得全部样本的标签向量
        '''
        content = self.get_file_content()
        labels = []
        for index in range(self.count):
            labels.append(self.norm(content[index + 8]))
        return labels

    def norm(self, label):
        '''
        内部函数，将一个值转换为10维标签向量
        '''
        label_vec = []
        label_value = label
        for i in range(10):
            if i == label_value:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)
        return label_vec

def get_training_data_set():
    '''
    获得训练数据集
    '''
    image_loader = ImageLoader('train-images.idx3-ubyte', 60000)
    label_loader = LabelLoader('train-labels.idx1-ubyte', 60000)
    return image_loader.load(), label_loader.load()

def get_test_data_set():
    '''
    获得测试数据集
    '''
    image_loader = ImageLoader('t10k-images.idx3-ubyte', 10000)
    label_loader = LabelLoader('t10k-labels.idx1-ubyte', 10000)
    return image_loader.load(), label_loader.load()

def get_result(vec):
    max_value_index = 0
    max_value = 0
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index

def evaluate(network, test_data_set, test_labels):
    error = 0
    total = len(test_data_set)
    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(network.predict(test_data_set[i]))
        if label != predict:
            error += 1
    return float(error) / float(total)

def train_and_evaluate():
    last_error_ratio = 1.0
    epoch = 0
    train_data_set, train_labels = get_training_data_set()
    test_data_set, test_labels = get_test_data_set()
    print('训练集和测试集获取完毕！')
    network = Network([784, 89, 10])
    print('神经网络构建完毕！')
    while True:
        epoch += 1
        network.train(train_labels, train_data_set, 0.3, 1)
        print('%s epoch %d finished' % (datetime.now(), epoch))
        if epoch % 5 == 0:
            error_ratio = evaluate(network, test_data_set, test_labels)
            print('%s after epoch %d, error ratio is %f' % (datetime.now(), epoch, error_ratio))
            if error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio

def gradient_check_test():
    net = Network([2, 2, 2])
    sample_feature = [0.9, 0.1]
    sample_label = [0.9, 0.1]
    gradient_check(net, sample_feature, sample_label)

if __name__ == '__main__':
    # gradient_check_test()
    train_and_evaluate()















