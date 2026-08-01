import tensorflow as tf


class Proj(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.w = self.add_weight("w", shape=(input_shape[-1], 3))

    def call(self, inputs):
        return tf.matmul(inputs, self.w)


class Gate(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.input_spec = tf.keras.layers.InputSpec(ndim=len(input_shape))

    def call(self, inputs):
        return tf.multiply(inputs, 2.0)


class Outer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.inner = Proj()

    def call(self, inputs):
        return self.inner(inputs)


a = tf.ones((2, 4))
b = tf.ones((2, 5))

p1 = Proj()
p2 = Proj()
assert p1(a).shape == (2, 3)
assert p2(b).shape == (2, 3)

g1 = Gate()
g2 = Gate()
assert g1(a).shape == (2, 4)
assert g2(b).shape == (2, 5)

o1 = Outer()
o2 = Outer()
assert o1(a).shape == (2, 3)
assert o2(b).shape == (2, 3)
