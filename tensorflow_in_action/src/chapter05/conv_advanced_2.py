#!/usr/local/anaconda2/envs/AI/bin/python 
import sys
sys.path.append('../cifar10')
import cifar10,cifar10_input
import tensorflow as tf
import math
import numpy as np
import time


# 参数定义
max_steps = 100000
batch_size = 128
learning_rate = 0.1
decay_step = 1000
data_dir = '/home/liujian/dataset/cifar10_data/cifar-10-batches-bin'

def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

# 下载并解压展开到默认位置tmp/cifar10_data
# cifar10.maybe_download_and_extract()

# 生成训练数据
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                            batch_size=batch_size)

# 生成测试数据
images_test, labels_test = cifar10_input.inputs(eval_data=True,
                                                data_dir=data_dir,
                                                batch_size=batch_size)

# 待接收数据
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

# 第一层：卷积层 + 池化层 + LRN(local response normalization)层
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, strides=[1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# 第二层：卷积层 + LRN层 + 池化层
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, strides=[1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 第三层：全连接层FCN
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=4e-2, wl=4e-3)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
fcn3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

# 第四层：全连接层FCN
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=4e-2, wl=4e-3)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
fcn4 = tf.nn.relu(tf.matmul(fcn3, weight4) + bias4)

# 第五层：全连接层FCN
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1/192.0, wl=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(fcn4, weight5), bias5)

# 定义损失函数
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                   labels=labels,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


# 计算损失并使用优化器优化
loss = loss(logits, label_holder)
global_step = tf.Variable(tf.constant(1), trainable=False)
lr = tf.subtract(learning_rate, tf.multiply(tf.cast(tf.multiply(tf.floordiv(global_step, decay_step), decay_step), tf.float32), 99e-8))
# lr = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step, decay_steps=decay_step, decay_rate=decay_rate, staircase=True)
train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step)

# 返回类型为布尔值，输出结果中topk的准确率，这里输出top1即分数最高的那一类的准确率
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

# 定义会话
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 启动图片数据增强的线程队列
tf.train.start_queue_runners()

# 训练
for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    lrt, _, loss_value = sess.run([lr, train_op, loss], feed_dict={image_holder: image_batch, label_holder: label_batch})
    duration = time.time() - start_time
    if step % 100 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)

        format_str = ('step %d,loss=%.2f, learning_rate=%.5f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value, lrt, examples_per_sec, sec_per_batch))

# 预测评估
num_examples = 10000
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, lable_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch, label_holder: lable_batch})
    true_count += np.sum(predictions)
    step += 1
precision = true_count / total_sample_count
print('precision @ SGD, 10k epoch = %.3f' % precision)

