import tensorflow as tf


# A base the engine resolves but PyDev's last segment does not name (#1099): PyDev
# sees only `Sequential`, while the engine follows `tf.keras.Sequential` to its
# summary class, whose superclass chain reaches `Model`. Keras supplies `training`
# from `Layer.__call__`, so the signature must be withheld.
class Stack(tf.keras.Sequential):
    def call(self, x, training=False):
        return tf.nn.relu(x)


y = Stack()(tf.random.normal([1, 3]))
assert y.shape == (1, 3)
