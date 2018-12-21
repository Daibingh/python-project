# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

if __name__ == "__main__":
    print("\n"*10)

    
    mnist = input_data.read_data_sets("./temp", one_hot=False)
    x_train, y_train = mnist.train.next_batch(5000)
    x_test, y_test = mnist.test.next_batch(100)

    # inoput
    X = tf.placeholder(tf.float32, shape=[None, 784])
    Y = tf.placeholder(tf.float32, shape=[None])
    Xt = tf.placeholder(tf.float32, shape=[784])

    # output
    # distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.add(X, tf.negative(Xt)), 2), 1))
    distance = tf.sqrt(tf.reduce_sum((X-Xt)**2, 1))

    index_min = tf.argmin(distance, 0)
    Yt = Y[index_min]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(x_test.shape[0]):
            x_t = x_test[i,:]
            y_true = y_test[i]
            y_t = sess.run(Yt, feed_dict={X: x_train, Y:y_train, Xt: x_t})
            print("number: {}, prediction is {}, truth is {}".format(i, y_t, y_true))