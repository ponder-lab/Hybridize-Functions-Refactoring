import tensorflow as tf


def reshape_to(y, n):
    return tf.reshape(y, [n, -1])


def count(c, flag):
    if flag:
        return 2
    return tf.shape(c)[0]


def pick(y, flag, c):
    n = count(c, flag)
    return reshape_to(y, n)


c = tf.ones([3, 5])
print(pick(tf.ones([2, 6]), True, c).shape)
print(pick(tf.ones([3, 4]), False, c).shape)
