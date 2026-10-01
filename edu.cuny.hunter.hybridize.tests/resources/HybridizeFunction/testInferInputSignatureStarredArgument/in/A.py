import tensorflow as tf


def t3(x, y):
    return x - y


def t4(x, y):
    return x - y


def t5(x, y):
    return x - y


t3(tf.ones([4]), *[tf.ones([4])])

t4(tf.ones([4]), tf.ones([4]))

t5(*[tf.ones([4])], y=tf.ones([4]))
