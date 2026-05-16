import json

import tensorflow as tf


def f(t):
    pass


shape = json.loads("[32]")
x = tf.keras.Input(shape=shape)
f(x)
