# The developer's own decorator order fails here: this program raises a TypeError when the method is called (#1000).
import tensorflow as tf


def make_grad(e):

    def grad(upstream):
        return upstream * (1 - 1 / (1 + e))

    return grad


class Ops:

    @tf.function
    @tf.custom_gradient
    def already(self, x):
        e = tf.exp(x)
        return tf.math.log(1 + e), make_grad(e)


Ops().already(tf.constant(100.0))
