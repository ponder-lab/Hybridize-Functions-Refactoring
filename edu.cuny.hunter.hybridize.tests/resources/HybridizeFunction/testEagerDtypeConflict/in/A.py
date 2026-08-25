import numpy as np
import tensorflow as tf

V32 = tf.Variable(tf.zeros([2]), name="v32")
V64 = tf.Variable(tf.zeros([2], dtype=tf.float64), name="v64")

X = np.array([1.5, 2.5])


def combine(x):
    return tf.reduce_sum(V32 * x), tf.reduce_sum(V64 * x)


def contracted(x):
    return tf.reduce_sum(tf.tensordot(V64, x, axes=1)), tf.reduce_sum(V32 * x)


# Eagerly the NumPy argument is coerced per operation: float32 at the V32 multiply and
# float64 at the V64 multiply, so both succeed. Any single input signature breaks one of
# the two, so hybridization must decline (#861 Case 1, the plural fallback).
combine(X)


# The same divergence with one side reached through a contraction rather than a multiply.
# `tensordot` does not convert its operands against each other; it requires them to agree,
# so a program that ran fixes the operand's dtype there just as firmly. Reading its
# contraction axes as the non-operand argument they are is what lets that side count at
# all: refused, the divergence goes unseen and a specification naming the fed dtype is
# emitted, which raises at the multiply (#909).
contracted(X)
