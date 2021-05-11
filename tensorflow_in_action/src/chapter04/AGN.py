import tensorflow as tf

class AdditiveGaussianNoiseAutoEncoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, 
                optimizer=tf.train.AdadeltaOptimizer(), scale=0.1):
        # 输入层节点数（自编码器的输出层与输入层节点数想等）
        self.n_input = n_input
        # 隐藏层节点数（这里仅设置一层隐藏层）
        self.n_hidden = n_hidden
        # 激活函数
        self.transfer = transfer_function  
        # 高斯噪声系数
        self.scale = tf.placeholder(tf.float32)
        self.train_scale = scale
        # 参数初始化
        network_weights = self._initialize_weights()
        self.weights = network_weights
        
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        # 输入到隐藏层的计算节点
        self.hidden = self.transfer(tf.add(tf.matmul(
                                    self.x + self.scale * tf.random_normal((n_input,)),
                                    self.weights['w1']), self.weights['b1']))
        # 隐藏层到输出层的计算节点
        self.reconstruction = tf.add(tf.matmul(self.hidden, 
                                    self.weights['w2']), self.weights['b2'])
        # 定义损失函数
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        # 定义优化目标
        self.optimizer = optimizer.minimize(self.cost)
        # 创建会话并初始化参数
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        
    def xavier_init(fan_in, fan_output, constant = 1):
        low = -constant * tf.sqrt(6.0 / (fan_in + fan_output))
        high = constant * tf.sqrt(6.0 / (fan_in + fan_output))
        return tf.random_uniform((fan_in, fan_output), minval=low, maxval=high, dtype=tf.float32)
    
    def _initialize_weights(self):
        all_weights = {}
        all_weights['w1'] = tf.Variable(self.xavier_init(self.n_input, 
                                                        self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], 
                                                dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,
                                 self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],
                                                dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer),
                feed_dict={self.x: X, self.scale: self.train_scale})
        return cost

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.train_scale})

    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.train_scale})

    def generate(self, hidden = None):
        if hidden == None:
            hidden = tf.random_normal((self.n_hidden,))
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.train_scale})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])




