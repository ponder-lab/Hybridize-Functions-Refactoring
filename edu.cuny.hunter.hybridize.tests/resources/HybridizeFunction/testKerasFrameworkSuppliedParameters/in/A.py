import tensorflow as tf


# Reduces TensorFlow2.0-Examples' RPN model (#881): `call` declares `training` with a
# default, no source call site passes it, and Keras's `Layer.__call__` supplies
# `training=False` itself on every invocation. An input signature covering only `x`
# raises on the first call ("got keyword argument `training` that was not included in
# input_signature"), so the signature must be withheld; the bare decorator runs.
class RPNplus(tf.keras.Model):
    def __init__(self):
        super(RPNplus, self).__init__()
        self.conv = tf.keras.layers.Conv2D(4, 3, padding="same")

    def call(self, x, training=False):
        return self.conv(x)


model = RPNplus()
image_data = tf.ones((2, 8, 8, 3))
y = model(image_data)
assert y.shape == (2, 8, 8, 4)
