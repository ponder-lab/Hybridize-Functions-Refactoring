import tensorflow as tf


# Keras models that build a functional model in their constructors and don't override
# `call`: applying an instance runs `Model.call` over the graph from `inputs` to
# `outputs`, so each consumer receives a (2, 4) float32 tensor.
class FunctionalKeyword(tf.keras.Model):
    def __init__(self):
        i = tf.keras.Input(shape=(3,))
        super().__init__(inputs=i, outputs=tf.keras.layers.Dense(4)(i))


class FunctionalPositional(tf.keras.Model):
    def __init__(self):
        i = tf.keras.Input(shape=(3,))
        super().__init__(i, tf.keras.layers.Dense(4)(i))


def consume_keyword(t):
    return t * 2.0


def consume_positional(t):
    return t * 2.0


def consume_direct(t):
    return t * 2.0


i = tf.keras.Input(shape=(3,))
direct = tf.keras.Model(inputs=i, outputs=tf.keras.layers.Dense(4)(i))

assert consume_keyword(FunctionalKeyword()(tf.ones([2, 3]))).shape == (2, 4)
assert consume_positional(FunctionalPositional()(tf.ones([2, 3]))).shape == (2, 4)
assert consume_direct(direct(tf.ones([2, 3]))).shape == (2, 4)
