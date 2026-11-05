import tensorflow as tf

DIMS = tf.constant([2, 3, 6])


def restore(x, i):
    return tf.reshape(x, [DIMS[i], -1])


t = tf.ones([2, 6])
print(restore(t, tf.constant(0)).shape)
print(restore(t, tf.constant(1)).shape)
