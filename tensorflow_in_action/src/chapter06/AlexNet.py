'''
ALexNet 
'''
import sys
sys.path.append('../')
from datetime import datetime
import math
import time
import tensorflow as tf

import config

# 参数设置
batch_size = 32
num_batches = 100

def print_activations(t):
    print(t.op.name, t.get_shape().as_list())

def inference(images):
    parameters = []

    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 96], 
                    dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, strides=[1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]
        lrn1 = tf.nn.lrn(conv1, depth_radius=4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn1')
        # padding模式为VALID,即取样时不能超过边框,不像SAME模式那样可以填充边界外的点
        pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], 
                                padding='VALID', name='pool1')
        print_activations(pool1)

    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 96, 256], 
                            dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), 
                            trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)                            
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv2)
        lrn2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn2')
        pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], 
                                padding='VALID', name='pool2')
        print_activations(pool2)

    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 384],
                            dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), 
                            trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)

    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 384], 
                            dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                            trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv4)
    
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32,
                            stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                            trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv5)
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                            padding='VALID', name='pool5')
        print_activations(pool5)
        return pool5, parameters
    
    # with tf.name_scope('fcn1') as scope:
    #     reshape_pool5 = tf.reshape(pool5, [batch_size, -1])
    #     dim = reshape_pool5.get_shape()[1].value
    #     weights = tf.Variable(tf.truncated_normal([dim, 4096], dtype=tf.float32,
    #                         stddev=1e-1), name='weights')
    #     biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
    #                         trainable=True, name='biases')
    #     bias = tf.nn.bias_add(tf.matmul(reshape_pool5, weights), biases)
    #     fcn1 = tf.nn.relu(bias, name=scope)
    #     parameters += [weights, biases]
    #     print_activations(fcn1)
    #
    # with tf.name_scope('fcn2') as scope:
    #     weights = tf.Variable(tf.truncated_normal([4096, 4096], dtype=tf.float32,
    #                         stddev=1e-1), name='weights')
    #     biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
    #                         trainable=True, name='biases')
    #     bias = tf.nn.bias_add(tf.matmul(fcn1, weights), biases)
    #     fcn2 = tf.nn.relu(bias, name=scope)
    #     parameters += [weights, biases]
    #     print_activations(fcn2)
    #
    # with tf.name_scope('fcn3') as scope:
    #     weights = tf.Variable(tf.truncated_normal([4096, 1000], dtype=tf.float32,
    #                         stddev=1e-1), name='weights')
    #     biases = tf.Variable(tf.constant(0.0, shape=[1000], dtype=tf.float32),
    #                          trainable=True, name = 'biaes')
    #     fcn3 = tf.nn.bias_add(tf.matmul(fcn2, weights), biases)
    #     parameters += [weights, biases]
    #     print_activations(fcn3)


def time_tensorflow_run(session, target, info_string):
    """
    statistics of time consumption
    :param session: tf.Session()
    :param target: 运算目标
    :param info_string: 目标描述串
    :return:
    """
    # 训练预热轮数
    num_steps_burn_in = 10
    # 总时耗
    total_duration = 0.0
    # 总时耗平方
    total_duration_squared = 0.0

    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                      (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration

    # 计算每轮batch训练的平均时耗及标准差
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, num_batches, mn, sd))


def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,
                                               image_size,
                                               image_size,
                                               3],
                                              dtype=tf.float32,
                                              stddev=1e-1))
        pool5, parameters = inference(images)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        time_tensorflow_run(sess, pool5, 'forward')

        objective = tf.nn.l2_loss(pool5)
        grad = tf.gradients(objective, parameters)
        time_tensorflow_run(sess, grad, 'forward-backward')


if __name__ == '__main__':
    run_benchmark()
    
