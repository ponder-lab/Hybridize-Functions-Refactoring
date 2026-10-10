import tensorflow as tf


def zeta(x):
    return tf.reduce_sum(x)


class Model:
    def beta(self, x):
        return zeta(x)


def alpha(x):
    return Model().beta(x)


alpha(tf.ones(2))
