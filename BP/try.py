#-*- coding:utf-8 -*-
"""
@file: try.py
@version:v1.0
@software:PyCharm

@author: fenglong.zhao
@contact: fenglong.zhao@irootech.com
@time:2019-5-23
"""
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None], name='x')
y = tf.placeholder(tf.float32, shape=[None], name='y')
w = tf.Variable(tf.truncated_normal(shape=[2], mean=0, stddev=0.1))

# gl_v = tf.Variable(0, trainable=False)
gl = tf.train.get_global_step()
learning_rate = tf.train.exponential_decay(0.1, gl, 10, 2, staircase=False)
loss = tf.square(w * x + y)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=gl)






