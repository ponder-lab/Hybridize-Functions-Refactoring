# Test for wala/ML#1095: a layer whose `call` registers a loss with `self.add_loss` writes to the
# layer, directly and through an enclosing layer whose `call` calls it.
import tensorflow as tf


class Var(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.w = self.add_weight("w", shape=(input_shape[-1], 2))

    def call(self, inputs):
        self.add_loss(tf.reduce_sum(tf.square(self.w)))
        return tf.matmul(inputs, self.w)


class Outer(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.inner = Var()

    def call(self, inputs):
        return self.inner(inputs)


class Plain(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.w = self.add_weight("w", shape=(input_shape[-1], 2))

    def call(self, inputs):
        return tf.matmul(inputs, self.w)


class Vae(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(2)

    def call(self, inputs):
        out = self.dense(inputs)
        self.add_loss(tf.reduce_sum(tf.square(out)))
        return out


x = tf.random.normal((8, 3))

var = Var()
var(x)
assert len(var.losses) == 1

outer = Outer()
outer(x)
assert len(outer.losses) == 1

plain = Plain()
plain(x)
assert len(plain.losses) == 0

vae = Vae()
vae(x)
assert len(vae.losses) == 1
