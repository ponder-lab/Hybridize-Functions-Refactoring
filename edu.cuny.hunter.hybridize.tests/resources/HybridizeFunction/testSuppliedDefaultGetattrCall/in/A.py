# Fixture for #1323's closed world: besides a call the analysis sees, the method is also called
# through `getattr` with a string. If the analysis does not see that call, its argument stays,
# and once the signature is on it raises `TypeError`.
import tensorflow as tf


class Layer:

    def f(self, x, axis=-1):
        return tf.reduce_sum(x, axis=axis)


layer = Layer()
layer.f(tf.constant([1.0, 2.0]), axis=-1)
getattr(layer, "f")(tf.constant([3.0, 4.0]), axis=-1)
