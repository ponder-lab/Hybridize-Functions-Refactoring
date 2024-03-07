# From https://github.com/tensorflow/tensorflow/issues/14359#issue-272179775

import tensorflow as tf


def f(a):
    pass


def g(a, b):
    model = tf.keras.models.Model(a, b)
    model(a)
    return model


a = tf.keras.layers.Input(shape=(64,))
b = tf.keras.layers.Dense(5)(a)
model = g(a, b)

for i in model.trainable_weights:
    f(i)
