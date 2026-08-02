# Fixture for the symbolic-iteration precondition (#830, distilling EventSeq.from_array):
# iterating a tensor parameter works eagerly (the elements are tensors) but raises
# OperatorNotAllowedInGraphError under tracing, so the conversion must be declined.
# An in-body tf.range loop is AutoGraph-supported and remains convertible.
import tensorflow as tf


def iterate_param(x):
    total = tf.zeros(())
    for e in x:
        total = total + e
    return total


def range_param_bound(x):
    acc = tf.reduce_sum(x)
    for i in tf.range(tf.shape(x)[0]):
        acc = acc + tf.cast(i, tf.float32)
    return acc


def range_loop(x):
    acc = tf.reduce_sum(x)
    for i in tf.range(2):
        acc = acc + tf.cast(i, tf.float32)
    return acc


def list_iter(xs, y):
    acc = y
    for t in xs:
        acc = acc + t
    return acc


t = tf.constant([1.0, 2.0, 3.0])
assert float(iterate_param(t)) == 6.0
assert float(range_loop(t)) == 7.0
assert float(range_param_bound(t)) == 9.0
assert float(tf.reduce_sum(list_iter([t, t], t))) == 18.0
