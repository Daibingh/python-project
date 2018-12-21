# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == "__main__":                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    print("\n"*10)

    # load data
    mnist = input_data.read_data_sets("./tmp", one_hot=True)

    # input variables
    sess = tf.InteractiveSession()
    X = tf.placeholder(tf.float32, shape=[None, 784])
    Y = tf.placeholder(tf.float32, shape=[None, 10])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   

    # Variables
    W = tf.Variable(np.random.randn(784,10)/784**(.5), dtype=tf.float32)
    b = tf.Variable(np.random.randn(10), dtype=tf.float32)
    
    Y_pre = tf.nn.softmax(tf.matmul(X, W)+b)


    loss = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(Y_pre)+(1-Y)*tf.log(1-Y_pre), 1))

    f = tf.equal(tf.argmax(Y_pre, 1), tf.argmax(Y, 1))
    acc = tf.reduce_mean(tf.cast(f, tf.float32))

    optimizer = tf.train.GradientDescentOptimizer(.01).minimize(loss)

    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        x_train, y_train = mnist.train.next_batch(50)
        sess.run(optimizer, feed_dict={X: x_train, Y: y_train})
        if i%100==0:
            ac, los = sess.run([acc, loss], feed_dict={X: mnist.test.images, Y: mnist.test.labels})
            print("iter: ", i, "acc=", ac, "loss=", los)
    sess.close()