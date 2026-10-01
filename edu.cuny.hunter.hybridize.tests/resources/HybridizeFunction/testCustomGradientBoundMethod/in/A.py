import tensorflow as tf


def make_grad(e):

    def grad(upstream):
        return upstream * (1 - 1 / (1 + e))

    return grad


class Ops:

    @tf.custom_gradient
    def bound(self, x):
        e = tf.exp(x)
        return tf.math.log(1 + e), make_grad(e)

    @classmethod
    @tf.custom_gradient
    def class_bound(cls, x):
        e = tf.exp(x)
        return tf.math.log(1 + e), make_grad(e)

    @staticmethod
    @tf.custom_gradient
    def unbound(x):
        e = tf.exp(x)
        return tf.math.log(1 + e), make_grad(e)


@tf.custom_gradient
def plain(x):
    e = tf.exp(x)
    return tf.math.log(1 + e), make_grad(e)


x = tf.constant(100.0)
ops = Ops()
ops.bound(x)
Ops.class_bound(x)
Ops.unbound(x)
plain(x)
