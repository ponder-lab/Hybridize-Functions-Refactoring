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


l = []


@tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.int32)])
def se(n):
    # A Python side-effect: de-hybridizing would change when it runs.
    l.append(n)
    return n


@tf.function(input_signature=[tf.TensorSpec(shape=[3], dtype=tf.float32)])
def se_barren(x):
    l.append(1)
    return x


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
    ]
)
def rec(x, n):
    if n == 0:
        return x
    return rec(x, n - 1)


declared(3)
se(3)
se_barren(tf.constant([1.0, 2.0, 3.0]))
rec(tf.constant(1.0), 3)
undeclared(3)
sig_barren(tf.constant([1.0, 2.0, 3.0]))
barren(tf.constant([1.0, 2.0, 3.0]))
sig_zero()
e = Exporter()
e.get_step()
