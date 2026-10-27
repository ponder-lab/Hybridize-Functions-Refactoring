import tensorflow as tf


class C:
    @staticmethod
    def s(x):
        return tf.shape(x)[1:]

    @staticmethod
    def restore(y, shape):
        return tf.reshape(y, shape)


o = C()


def trailing(x):
    return o.s(x)


def flattened(x):
    return o.restore(x, [-1])


t = tf.ones([2, 3, 4])
print(tf.reshape(t, tf.concat(([-1], trailing(t)), 0)).shape)
print(flattened(t).shape)
