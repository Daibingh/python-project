# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os, os.path
import matplotlib.pyplot as plt
import numpy as np

acc_ = []
loss_ = []
iter = []

def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)

def main():
    data = input_data.read_data_sets('F:\\Python\\deeplearning\\' \
                                     'tensorflow_examples\\examples\\' \
                                     'tutorials\\mnist\\temp', one_hot=True)

    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
    with tf.name_scope('reshape1'):
        xx = tf.reshape(x, [-1, 28, 28, 1])

    with tf.name_scope('conv1'):
        init = tf.truncated_normal([5, 5, 1, 20], mean=0, stddev=.1)
        w_conv1 = tf.Variable(init)
        b_conv1 = tf.Variable(tf.constant(.1, shape=[20]))
        h_conv1 = tf.nn.relu(tf.nn.conv2d(xx, w_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1)

    with tf.name_scope('max_pool1'):
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    with tf.name_scope('reshape2'):
        h_pool1_flatten = tf.reshape(h_pool1, [-1, 14 * 14 * 20])

    with tf.name_scope('fc1'):
        init = tf.truncated_normal([14*14*20, 100], mean=0, stddev=.1)
        w_fc1 = tf.Variable(init)
        b_fc1 = tf.Variable(tf.constant(.1, shape=[100]))
        h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flatten, w_fc1) + b_fc1)

    with tf.name_scope('output'):
        init = tf.truncated_normal([100, 10], mean=0, stddev=.1)
        w_output = tf.Variable(init)
        b_output = tf.Variable(tf.constant(.1, shape=[10]))
        y = tf.nn.softmax(tf.matmul(h_fc1, w_output) + b_output)

    with tf.name_scope('loss'):
        loss = - tf.reduce_mean(y_*tf.log(y) + (1-y_)*tf.log(1-y))

    with tf.name_scope('optimization'):
        optimize_op = tf.train.AdamOptimizer(.0001).minimize(loss)

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32), name='accuracy')

    with tf.name_scope('summary'):
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()

    if not os.path.exists('./logdir'):
        os.makedirs('./logdir')
    del_file('./logdir')
    writer = tf.summary.FileWriter('./logdir')
    writer.add_graph(tf.get_default_graph())

    # print(os.path.abspath('.') + '\\logdir')
    # os.system('tensorboard --logdir=' + os.path.abspath('.') + '\\logdir')
    # return

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000*5):
            if i % 100 == 0:
                acc, mg, loss1 = sess.run([accuracy, merged, loss], feed_dict={x: data.test.images, y_: data.test.labels})
                writer.add_summary(mg, i)
                print('NO.', i, ':', acc)
                iter.append(i)
                acc_.append(acc)
                loss_.append(loss1)
            batch = data.train.next_batch(50)
            sess.run([optimize_op], feed_dict={x: batch[0], y_: batch[1]})
    fig1 = plt.figure()
    plt.plot(np.array(iter), np.array(acc_))
    fig2 = plt.figure()
    plt.plot(np.array(iter), np.array(loss_))
    fig1.savefig('aa.png')
    fig2.savefig('bb.png')


if __name__ == '__main__':
    main()
