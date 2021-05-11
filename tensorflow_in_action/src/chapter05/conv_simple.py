from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from config import MNIST_PATH

mnist = input_data.read_data_sets(MNIST_PATH, one_hot=True)
sess = tf.InteractiveSession()

# 权重初始函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 偏置初始函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积核操作定义
def conv2d(x, W):
    # strides定义卷积模板移动步长，都是1表明会划过图片中每一个点
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 最大池化函数
def max_pool_2x2(x):
    # ksize定义使用多大的池化([1,2,2,1]表示2*2的最大池化，将一个2*2的像素块降为1*1的像素块)
    # strides决定最终保留的图像大小，[1, 2, 2, 1]表示横竖两个方向以2为步长，得到的图像尺寸将缩小一半，都为1则大小不变
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 输入数据接收口
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])  # 转换成卷积能处理的2D图像(-1表示样本数量不定，1表示通道数为1)

# 第一层：卷积层
W_conv1 = weight_variable([5, 5, 1, 32]) # 参数：前2个参数5，5为卷积核大小5*5，第3个参数表示通道数，第4个参数表示卷积核数目
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层：卷积层
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 第三层：全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # 将第二层卷积层的输出张量舒展成1D向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 第四层：dropout层
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 第五层：softmax层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 定义损失函数和优化器
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 定义准确率评估方法
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 在训练集上进行训练和验证
tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# 在测试集上进行测试
print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))