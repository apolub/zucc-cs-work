import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import time
import math
import numpy as np
import matplotlib.pyplot as plt
 

#数据集下载
cifar10 = tf.compat.v1.keras.datasets.cifar10
(images_train, labels_train),(images_test, labels_test) = cifar10.load_data()

# move warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
 
# 训练参数
BATCH_SIZE = 128
TRAINING_STEPS = 5000
 
# 指数衰减学习率参数
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.9
 
 
# 定义初始化权值函数
def variable_with_weight_loss(shape, stddev, w1):
    var = tf.compat.v1.Variable(tf.compat.v1.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        # w1控制L2正则化的大小
        weight_loss = tf.multiply(tf.compat.v1.nn.l2_loss(var), w1, name='weight_loss')
        tf.compat.v1.add_to_collection('losses', weight_loss)
    return var
 
 
 
# 输入层
image_in = tf.compat.v1.placeholder(tf.compat.v1.float32, [BATCH_SIZE, 24, 24, 3])  # 裁剪后尺寸为24×24，彩色图像通道数为3
label_in = tf.compat.v1.placeholder(tf.compat.v1.int32, [BATCH_SIZE])
is_training = tf.compat.v1.placeholder(tf.compat.v1.bool, [])
 
# 卷积层1
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)  # 5×5的卷积核，3个通道，64个滤波器
kernel1 = tf.compat.v1.nn.conv2d(image_in, weight1, strides=[1, 1, 1, 1], padding='SAME')
bias1 = tf.compat.v1.Variable(tf.compat.v1.constant(0.1, shape=[64]))
active1 = tf.compat.v1.nn.bias_add(kernel1, bias1)
bn1 = tf.compat.v1.layers.batch_normalization(active1, training=is_training)
conv1 = tf.compat.v1.nn.relu(bn1)
pool1 = tf.compat.v1.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
 
# 卷积层2
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)  # 5×5的卷积核，第一个卷积层输出64个通道，64个滤波器
kernel2 = tf.compat.v1.nn.conv2d(pool1, weight2, strides=[1, 1, 1, 1], padding='SAME')
bias2 = tf.compat.v1.Variable(tf.constant(0.1, shape=[64]))
active2 = tf.compat.v1.nn.bias_add(kernel2, bias2)
bn2 = tf.compat.v1.layers.batch_normalization(active2, training=is_training)
conv2 = tf.compat.v1.nn.relu(bn2)
pool2 = tf.compat.v1.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
 
# 全连接层1
reshape = tf.compat.v1.reshape(pool2, [BATCH_SIZE, -1])  # 将数据变为1D数据
dim = reshape.get_shape()[1].value  # 获取维度
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
bias3 = tf.compat.v1.Variable(tf.compat.v1.constant(0.1, shape=[384]))
local3 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(reshape, weight3) + bias3)
 
# 全连接层2
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
bias4 = tf.compat.v1.Variable(tf.compat.v1.constant(0.1, shape=[192]))
local4 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(local3,  weight4) + bias4)
 
# 输出层
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1/192.0, w1=0.0)
bias5 = tf.compat.v1.Variable(tf.compat.v1.constant(0.0, shape=[10]))
logits = tf.compat.v1.add(tf.compat.v1.matmul(local4, weight5), bias5)
 
 
# 计算loss
def loss(logits, labels):
    labels = tf.compat.v1.cast(labels, tf.compat.v1.int64)
    cross_entropy = tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.compat.v1.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.compat.v1.get_collection('losses'), name='total_loss')
 
 
# training operation
global_step = tf.compat.v1.Variable(0, trainable=False)
learning_rate = tf.compat.v1.train.exponential_decay(LEARNING_RATE_BASE, global_step, int(5e4/BATCH_SIZE), LEARNING_RATE_DECAY)
loss = loss(logits, label_in)
update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
with tf.compat.v1.control_dependencies(update_ops):  # 保证train_op在update_ops执行之后再执行
    train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
top_k_op = tf.compat.v1.nn.in_top_k(logits, label_in, 1)  # 得分最高的那一类的准确率
sess = tf.compat.v1.InteractiveSession()
tf.compat.v1.global_variables_initializer().run()
 
# 启动线程，使用16个线程来加速
tf.compat.v1.train.start_queue_runners()
 
# 训练
losss = []
steps = []
rates = []
for step in range(TRAINING_STEPS):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value, _, now_learn_rate = sess.run([train_op, loss, global_step, learning_rate],
                                                feed_dict={image_in: image_batch, label_in: label_batch,
                                                           is_training: True})
    duration = time.time() - start_time  # 运行时间
    if step % 20 == 0:
        losss.append(loss_value)
        steps.append(step)
        rates.append(now_learn_rate)
        sec_per_batch = float(duration)  # 每个batch的时间
        format_str = 'step %d, loss=%.2f(%.3f sec/batch), learning_rate=%f'
        print(format_str % (step, loss_value, sec_per_batch, now_learn_rate))
 
# 测试模型准确率
num_examples = 10000
num_iter = int(math.ceil(num_examples / BATCH_SIZE))
true_count = 0
total_sample_count = num_iter * BATCH_SIZE
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={image_in: image_batch, label_in: label_batch, is_training: False})
    true_count += np.sum(predictions)
    step += 1
precision = true_count / total_sample_count
print('precision: %.3f' % precision)
 
# learning rate
plt.figure()
plt.plot(steps, rates)
plt.xlabel('Number of steps')
plt.ylabel('Learning rate')
 
# loss
plt.figure()
plt.plot(steps, losss)
plt.xlabel('Number of steps')
plt.ylabel('Loss')
plt.show()
 
tf.logging.set_verbosity(old_v)