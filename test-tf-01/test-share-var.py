import tensorflow as tf
import numpy as np


def fun():
    with tf.variable_scope("aaa", reuse=True):
        bb = tf.get_variable("b")
        print(bb.name, bb.eval())

print("\n"*10)
with tf.variable_scope("aaa") as scope:
    sess = tf.InteractiveSession()
    init = tf.constant_initializer(value=3)
    a = tf.get_variable("a", shape=[1])
    b = tf.get_variable("b", shape=[1], initializer=init)
    c = tf.Variable(5, name="c")
    # scope.reuse_variables()
    # bb = tf.get_variable("b", [1], initializer=tf.constant_initializer(value=4))
    # cc = tf.Variable(10, name="c")
    

sess.run(tf.global_variables_initializer())
print(b.name, b.eval())
fun()init