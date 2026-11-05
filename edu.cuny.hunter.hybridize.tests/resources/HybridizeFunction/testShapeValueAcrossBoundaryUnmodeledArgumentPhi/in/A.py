import tensorflow as tf


def reshape_to(y, n):
    return tf.reshape(y, [n, -1])


def pick(y, flag, c):
    n = 2 if flag else tf.shape(c)[0]
    return reshape_to(y, n)


c = tf.ones([3, 5])
print(pick(tf.ones([2, 6]), True, c).shape)
print(pick(tf.ones([3, 4]), False, c).shape)
