# -*- coding: utf-8 -*-
# @Time    : 19-1-24 下午9:50
# @Author  : liujian
# @File    : MLP.py
# @Software: PyCharm

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from config import MNIST_PATH

mnist = input_data.read_data_sets(MNIST_PATH, one_hot=True)
sess = tf.InteractiveSession()
# 输入层节点数
in_units = 784
# 隐藏层节点数
h1_units = 300

# 参数
# 截断正态分布，采样值到均值距离不超过2倍标准差
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

# 输入数据x
x = tf.placeholder(tf.float32, [None, in_units])
# dropout过程中节点保留概率
keep_prob = tf.placeholder(tf.float32)

# ReLU缓解梯度弥散问题
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
# dropout避免过拟合
hidden1_dropout = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_dropout, W2) + b2)

# 定义损失函数
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# 定义优化器及优化目标(Adagrad自适应学习速率)
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

# 初始化参数并训练
tf.global_variables_initializer().run()
for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})

# 预测
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
