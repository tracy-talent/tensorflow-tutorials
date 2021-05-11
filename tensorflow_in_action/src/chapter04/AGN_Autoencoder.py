# -*- coding: utf-8 -*-
# @Time    : 19-1-24 下午3:00
# @Author  : liujian
# @File    : AGN_Autoencoder.py.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from sklearn.preprocessing import StandardScaler
import numpy as np

from AGN import AdditiveGaussianNoiseAutoEncoder
from config import MNIST_PATH

def standard_scale(X_train, X_test):
    preprocessor = StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size + 1)
    return data[start_index:start_index + batch_size]

def main():
    mnist = input_data.read_data_sets(MNIST_PATH, one_hot=True)
    X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

    # 可调参数
    n_samples = int(mnist.train.num_examples)
    training_epochs = 20
    batch_size = 128
    display_step = 1
    n_hidden = 200

    autoencoder = AdditiveGaussianNoiseAutoEncoder(n_input = 784,
                                                   n_hidden = n_hidden,
                                                   transfer_function = tf.nn.softplus,
                                                   optimizer = tf.train.AdamOptimizer(learning_rate = 0.001),
                                                   scale = 0.01)
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = n_samples // batch_size
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, batch_size)

            cost = autoencoder.partial_fit(batch_xs)
            avg_cost += cost / n_samples * batch_size

        if epoch % display_step == 0:
            print("Epoch:", "%04d" % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("total cost:", autoencoder.calc_total_cost(X_test))

if __name__ == '__main__':
    main()