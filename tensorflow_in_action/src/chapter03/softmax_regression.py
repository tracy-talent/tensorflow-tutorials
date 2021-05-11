from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from config import MNIST_PATH

mnist = input_data.read_data_sets(MNIST_PATH, one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.images.shape)

sess = tf.InteractiveSession()
# 待feed的输入数据
x = tf.placeholder(tf.float32, [None, 784])
# Variable类型的数值保存在显存中不会丢失
# 训练参数W:像素值权重
W = tf.Variable(tf.zeros([784, 10]))
# 训练参数b:数据倾向bias
b = tf.Variable(tf.zeros([10]))
# 定义算法公式构建计算图，softmax regression训练得到预测标签
y = tf.nn.softmax(tf.matmul(x, W) + b)
# 数据的实际标签
y_ = tf.placeholder(tf.float32, [None, 10])
# 定义损失函数：交叉熵
# 交叉熵作损失函数好处是使用sigmoid函数在梯度下降时能避免均方误差损失函数学习速率降低的问题，因为学习速率被输出的误差所控制
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# 定义优化器：梯度下降
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 全局参数初始化
tf.global_variables_initializer().run()

# 迭代训练
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

# 预测
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))

