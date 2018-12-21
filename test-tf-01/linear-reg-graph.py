# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("\n"*10)

    rng = np.random
    learn_rate = .01
    training_epochs = 100
    display_step = 10
    train_X = np.asarray([5.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
    train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
    n_samples = train_X.shape[0]

    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    w = tf.Variable(np.random.randn())
    b = tf.Variable(np.random.randn())

    # pred = tf.add(tf.multiply(X, w), b)
    pred = w*X+b
    cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
    optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(training_epochs):
            sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})
            if i%display_step == 0:
                c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
                print("epoch:", i, "cost=", c, "w=", sess.run(w), "b=", sess.run(b))
        print("train finish", "cost=", c, "w=", sess.run(w), "b=", sess.run(b))

        plt.plot(train_X, train_Y, 'ro', label="origin data")
        plt.plot(train_X, train_X*sess.run(w)+sess.run(b), label="fitted line")
        plt.legend()
        plt.show()
