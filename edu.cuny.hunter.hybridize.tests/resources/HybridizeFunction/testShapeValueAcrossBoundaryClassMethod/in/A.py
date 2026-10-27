import tensorflow as tf


class C:
    @classmethod
    def k(cls, x):
        return tf.shape(x)[1:]


def trailing(x):
    return C.k(x)


t = tf.ones([2, 3, 4])
print(tf.reshape(t, tf.concat(([-1], trailing(t)), 0)).shape)
