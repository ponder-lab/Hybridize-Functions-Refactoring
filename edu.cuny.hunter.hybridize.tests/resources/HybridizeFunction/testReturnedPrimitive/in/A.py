# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1117

import tensorflow as tf


def hidden_units(x):
    x = tf.abs(x)
    return 6


def grid(x):
    x = tf.abs(x)
    return 2, 3


def is_training(x):
    x = tf.abs(x)
    return True


class Head(tf.keras.layers.Layer):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def call(self, inputs):
        return tf.keras.layers.Dense(units=hidden_units(self.scale))(inputs)


class Grid(tf.keras.layers.Layer):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def call(self, inputs):
        rows, cols = grid(self.scale)
        return tf.keras.layers.Reshape((rows, cols))(inputs)


t = tf.constant([1.0, 2.0])

names = {hidden_units(t): "hidden"}
print(names[6])

rows, cols = grid(t)
print({(rows, cols): "grid"}[(2, 3)])

modes = {True: "train", False: "eval"}
print(modes[is_training(t)])

inputs = tf.keras.Input(shape=(6,))
model = tf.keras.Model(inputs, Grid(t)(Head(t)(inputs)))
print(model.output_shape)
