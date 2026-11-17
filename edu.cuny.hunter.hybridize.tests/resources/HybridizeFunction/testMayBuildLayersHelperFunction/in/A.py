import os
import tempfile

import tensorflow as tf


def cnn_stack(layers, inputs):
    next_input = inputs

    for i, layer in enumerate(layers):
        outputs = layer(next_input)
        if i > 0:
            outputs += next_input
        next_input = tf.tanh(outputs)

    return next_input


class Encoder(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.cnn_a = [tf.keras.layers.Conv1D(4, 3, padding="same") for _ in range(2)]
        self.cnn_c = [tf.keras.layers.Conv1D(4, 3, padding="same") for _ in range(2)]

    def call(self, inputs):
        cnn_a = cnn_stack(self.cnn_a, inputs)
        cnn_c = cnn_stack(self.cnn_c, inputs)
        return cnn_a, tf.reduce_mean(cnn_c, axis=1)


def apply(layer, x):
    return layer(x)


def apply_one(layer, x):
    return layer(x)


def apply_each(layer, x):
    return layer(x)


def apply_last(x, layer):
    return layer(x)


def apply_if(layer, x, flag):
    if flag:
        x = layer(x)

    return x


def stack_one(layers, x):
    for layer in layers:
        x = layer(x)

    return x


def construct(x):
    return tf.keras.layers.Dense(3)(x)


def train_step(model, x):
    return model(x)


class Tools:

    @staticmethod
    def run(layer, x):
        return layer(x)

    @classmethod
    def run_class(cls, layer, x):
        return layer(x)


class Named(tf.keras.layers.Layer):

    def __init__(self, label):
        super().__init__()
        self.dense = tf.keras.layers.Dense(3, name=label + "_dense")

    def call(self, x):
        return self.dense(x)


def run_named(layer, x):
    return layer(x)


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(3)

    def call(self, x):
        return self.dense(x)


Encoder()(tf.zeros([2, 5, 4]))

first = tf.keras.layers.Dense(3)
second = tf.keras.layers.Dense(3)
apply(first, tf.ones([2, 4]))
apply(second, tf.ones([2, 4]))

only = tf.keras.layers.Dense(3)
apply_one(only, tf.ones([2, 4]))
apply_one(only, tf.ones([2, 4]))

for _ in range(2):
    fresh = tf.keras.layers.Dense(3)
    apply_each(fresh, tf.ones([2, 4]))

last = tf.keras.layers.Dense(3)
apply_last(tf.ones([2, 4]), last)
apply_last(tf.zeros([2, 4]), last)

gated = tf.keras.layers.Dense(3)
apply_if(gated, tf.ones([2, 4]), False)
apply_if(gated, tf.ones([2, 4]), True)

pair = [tf.keras.layers.Dense(4) for _ in range(2)]
stack_one(pair, tf.ones([2, 4]))
stack_one(pair, tf.ones([2, 4]))

construct(tf.ones([2, 4]))

net = Net()
train_step(net, tf.ones([2, 4]))
train_step(net, tf.ones([2, 4]))

static_first = tf.keras.layers.Dense(3)
static_second = tf.keras.layers.Dense(3)
Tools.run(static_first, tf.ones([2, 4]))
Tools.run(static_second, tf.ones([2, 4]))

class_first = tf.keras.layers.Dense(3)
class_second = tf.keras.layers.Dense(3)
Tools.run_class(class_first, tf.ones([2, 4]))
Tools.run_class(class_second, tf.ones([2, 4]))

named = Named("a")
run_named(named, tf.ones([2, 4]))

net.save_weights(os.path.join(tempfile.mkdtemp(), "t.h5"))
