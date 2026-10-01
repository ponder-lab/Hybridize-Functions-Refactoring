import pickle

import numpy as np
import tensorflow as tf

V = tf.Variable(tf.zeros([2]), name="weight")

X64 = np.array([1.5, 2.5])

with open("batch.pkl", "rb") as f:
    batch = pickle.load(f)


def pin_mix(x, pair):
    a, b = pair
    return tf.reduce_sum(V * x) + a + b


def pin_typed(x, pair):
    a, b = pair
    return tf.reduce_sum(V * x) + a + b


def flat_and_cont(x, pair):
    a, b = pair
    return x + a + b


pin_mix(X64, (tf.zeros([2]), tf.zeros([2])))
pin_mix(X64, batch["pair"])

pin_typed(X64, (tf.zeros([2]), tf.zeros([2])))

flat_and_cont(tf.zeros([2]), (tf.zeros([2]), tf.zeros([2])))
flat_and_cont(batch["x"], batch["pair"])
