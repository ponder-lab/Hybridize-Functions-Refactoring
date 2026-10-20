import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Input

# Reduces transformer's `text_classification.py` (#1110): model code factored into a
# plain class, not a Keras layer, whose `__call__` is applied to an in-frame `Input` by
# calling an instance, `Encoder(args)(inp)`. `tf.function` is one of the APIs a
# `KerasTensor` refuses, so decorating `Encoder.__call__` raises a `TypeError` on the
# first call, before anything is traced. `Scaler.__call__` is applied the same way to a
# real tensor, which leaves it convertible.


class Encoder:

    def __init__(self, vocab_size, d_model):
        self.embedding = Embedding(vocab_size, d_model)
        self.dense = Dense(d_model)

    def __call__(self, x):
        return self.dense(self.embedding(x))


class Scaler:

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return x * self.factor


inp = Input((None,), dtype="int32", name="inp")
net = Encoder(vocab_size=100, d_model=8)(inp)
model = tf.keras.Model(inputs=inp, outputs=net)
assert model(tf.ones((2, 3), dtype=tf.int32)).shape == (2, 3, 8)

assert Scaler(2.0)(tf.ones((2, 4))).shape == (2, 4)
