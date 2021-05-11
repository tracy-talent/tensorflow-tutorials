'''
@Author:  liujian
@Email:  brooksj@foxmail.com
@File:  PolicyNetWork.py
@DateTime:  2019/10/21 20:49:27
'''

# policy network for catpole problem
# here put the import lib
import numpy as np
import tensorflow as tf
import gym

# # 随机策略
env = gym.make('CartPole-v0')
# env.reset()
# random_episodes = 0
# reward_sum = 0
# while random_episodes < 10: # 进行十次随机测试， 每次测试后重置环境
#     env.render()
#     observation, reward, done, _ = env.step(np.random.randint(0, 2))
#     reward_sum += reward
#     if done:
#         random_episodes += 1
#         print('Reward for this episode was:', reward_sum)
#         reward_sum = 0
#         env.reset()
'''
经测试随机策略reward均值差不多在20~30
'''

# 策略网络-MLP
## 超参数
hidden_size = 50
batch_size = 25 # batch_zie等同于环境重置次数
learning_rate = 1e-1 
env_dim = 4 # 环境特征维度
gamma = 0.99 # reward discount ratio

observations = tf.placeholder(tf.float32, [None, env_dim], name='input_x')
W1 = tf.get_variable("W1", shape=[env_dim, hidden_size], \
    initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.get_variable("W2", shape=[hidden_size, 1], \
    initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)

adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad = tf.placeholder(tf.float32, name="batch_grad1")
W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
batch_grad = [W1Grad, W2Grad]
tvars = tf.trainable_variables()
updateGrads = adam.apply_gradients(zip(batch_grad, tvars))

def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

input_y = tf.placeholder(tf.float32, [None, 1], name='input_y')
advantages = tf.placeholder(tf.float32, [None, 1], name='reward_signal')
loglik = tf.log(input_y * probability + (1 - input_y) * (1 - probability))
loss = -tf.reduce_mean(loglik * advantages)
newGrads = tf.gradients(loss, tvars)

xs, ys, drs = [], [], []
reward_sum = 0
rendering = False
episode_number = 1
total_episodes = 10000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    observation = env.reset()

    gradBuffer = sess.run(tvars)
    for idx, grad in enumerate(gradBuffer):
        gradBuffer[idx] = grad * 0
    
    while episode_number <= total_episodes:
        if rendering == True or reward_sum / batch_size > 100:
            env.render()
            rendering = True
        x = np.reshape(observation, [1, env_dim])
        tfprob = sess.run(probability, feed_dict={observations:x})
        action = 1 if np.random.uniform() < tfprob else 0
        xs.append(x)
        ys.append(action)

        observation, reward, done, info = env.step(action)
        reward_sum += reward
        drs.append(reward)

        if done:
            episode_number += 1
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            xs, ys, drs = [], [], []

            discount_epr = discount_rewards(epr)
            discount_epr -= np.mean(discount_epr)
            discount_epr /= np.std(discount_epr)

            tGrad = sess.run(newGrads, feed_dict={observations:epx, \
                input_y:epy, advantages:discount_epr})
            for idx, grad in enumerate(tGrad):
                gradBuffer[idx] += grad
            
            if episode_number % batch_size == 0:
                sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
                for idx, grad in enumerate(gradBuffer):
                    gradBuffer[idx] = grad * 0
                print('Average reward for episode %d : %f.' % \
                    (episode_number, reward_sum / batch_size))
                if reward_sum / batch_size > 200:
                    print('Task solved in', episode_number, 'episodes!')
                    break
                reward_sum = 0
            observation = env.reset()
