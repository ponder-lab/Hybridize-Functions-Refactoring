import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.int32)])
def declared(n):
    # Called with a Python int, which the analysis doesn't see as a tensor; the signature converts it to one.
    return n + 1


@tf.function
def undeclared(n):
    return n + 1


@tf.function(input_signature=[tf.TensorSpec(shape=[3], dtype=tf.float32)])
def sig_barren(x):
    # A tensor parameter and no tensor computation: barren, but its signature pins the boundary.
    return x


@tf.function
def barren(x):
    return x


@tf.function(input_signature=[])
def sig_zero():
    return 1


class Exporter(tf.Module):
    def __init__(self):
        self.step = tf.Variable(0)

    @tf.function(input_signature=[])
    def get_step(self):
        return self.step


declared(3)
undeclared(3)
sig_barren(tf.constant([1.0, 2.0, 3.0]))
barren(tf.constant([1.0, 2.0, 3.0]))
sig_zero()
e = Exporter()
e.get_step()
