import numpy as np
import tensorflow as tf


# Receives the float64 concatenation of two `np.random.normal` arrays.
def total(x):
    return tf.reduce_sum(x)


# Receives the float64 concatenation of two outer products of `np.random.normal` vectors.
def outer_total(x):
    return tf.reduce_sum(x)


# Receives the float64 stack of two `np.random.normal` arrays.
def stacked_total(x):
    return tf.reduce_sum(x)


# Receives the rows `a[idx,]` selects, with `idx` from `np.random.choice`.
def selected_total(x):
    return tf.reduce_sum(x)


# Receives the int64 indices `np.random.choice` draws as `indices`.
def gather(params, indices):
    return tf.gather(params, indices)


a = np.random.normal(0.0, 1.0, (4, 3))
b = np.random.normal(0.0, 1.0, (4, 3))
total(np.concatenate((a, b), axis=0))

u = np.random.normal(0.0, 1.0, 4)
v = np.random.normal(0.0, 1.0, 3)
outer_total(np.concatenate((np.outer(u, v), np.outer(v, v)), axis=0))

stacked_total(np.stack((a, b), axis=2))

idx = np.random.choice(4, 2)
selected_total(a[idx,])
gather(tf.ones((4,)), idx)
