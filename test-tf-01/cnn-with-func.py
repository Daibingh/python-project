# -*- coding: utf-8 -*-

"""there is problem!!!!!!!!!!!!!!!!!!"""


import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def conv2d(x, w, b):
    return tf.nn.relu(tf.nn.conv2d(x, w, strides=[1,1,1,1], padding="SAME") + b)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding="SAME")

def conv_net(x, w, b, prob):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    h1_conv = conv2d(x, w["wc1"], b["bc1"])
    h1_pool = maxpool2d(h1_conv)
    h2_conv = conv2d(h1_pool, w["wc2"], b["bc2"])
    h2_pool = maxpool2d(h2_conv)
    
    h2_pool_flat = tf.reshape(h2_pool, [-1, 7*7*64])
    h3 = tf.nn.relu(tf.matmul(h2_pool_flat, w["wd1"]) + b["bd1"])
    h3 = tf.nn.dropout(h3, prob)
    return tf.nn.relu(tf.matmul(h3, w["out"]) + b["out"])


if __name__ == "__main__":
    print("\n"*10)

    mnist = input_data.read_data_sets("./temp", one_hot=True)

    # define super-params
    learning_rate = .01
    batch_size = 10
    disp_step = 20
    

    # input var
    X = tf.placeholder(tf.float32, shape=[None, 784])
    Y = tf.placeholder(tf.float32, shape=[None, 10])
    prob = tf.placeholder(tf.float32)

    # params
    weights = {
        "wc1": tf.Variable(tf.random_normal([5,5,1,32])),
        "wc2": tf.Variable(tf.random_normal([5,5,32,64])),
        "wd1": tf.Variable(tf.random_normal([7*7*64, 1024])),
        "out": tf.Variable(tf.random_normal([1024, 10]))
    }

    bias = {
        "bc1": tf.Variable(tf.random_normal([32])),
        "bc2": tf.Variable(tf.random_normal([64])),
        "bd1": tf.Variable(tf.random_normal([1024])),
        "out": tf.Variable(tf.random_normal([10]))
    }

    # compute prediction
    Y_pre = conv_net(X, weights, bias, prob)
    
    # compute loss
    loss = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(Y_pre)+(1-Y)*tf.log(1-Y_pre), 1))

    #compute accurate
    accurate = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y,1), tf.argmax(Y_pre,1)), tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(5000):
            x_train, y_train = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={X: x_train, Y: y_train, prob: .75})  
            if i % disp_step == 0:
                acc, los = sess.run([accurate, loss], feed_dict={X: mnist.test.images, Y: mnist.test.labels, prob: 1.0})
                print("i = {}, acc = {}, loss = {}".format(i, acc, los))
    

