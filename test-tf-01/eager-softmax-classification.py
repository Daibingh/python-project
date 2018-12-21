from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


if __name__ == "__main__":
    print("\n"*10)

    tf.enable_eager_execution()
    tfe = tf.contrib.eager

    # load minist dataset
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    
    # varialbes
    W = tfe.Variable(np.random.randn(784,10)/784**(.5), dtype=tf.float32)
    b = tfe.Variable(np.random.randn(10), dtype=tf.float32)

    def model(X):
        X = tf.convert_to_tensor(X)
        return tf.nn.softmax(tf.matmul(X, W)+b)

    def compute_loss(X, Y):
        Y_pre = model(X)
        return tf.reduce_mean(-tf.reduce_sum(Y*tf.log(Y_pre)+(1-Y_pre)*tf.log(1-Y_pre), 1))

    def compute_acc(Y_pre, Y):
        f = tf.equal(tf.argmax(Y_pre, 1), tf.argmax(Y, 1))
        return tf.reduce_mean(tf.cast(f, tf.float32))

    grad_fn = tfe.implicit_value_and_gradients(compute_loss)
    optimizer = tf.train.GradientDescentOptimizer(.01)

    for i in range(5000):
        x_train, y_train = mnist.train.next_batch(100)
        loss, grads = grad_fn(x_train, y_train)
        optimizer.apply_gradients(grads)
        if i % 100 ==0:
            Y_pre = model(mnist.test.images)
            acc = compute_acc(Y_pre, mnist.test.labels)
            print("iter: {}, acc = {}, loss={}".format(i, acc, loss))



    