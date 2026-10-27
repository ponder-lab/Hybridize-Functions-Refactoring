import tensorflow as tf


# The `mask` arm of the framework-supplied parameter rule (#881), which the `training`
# arm's fixture does not reach: `KERAS_FRAMEWORK_SUPPLIED_PARAMETER_NAMES` holds both
# names, and until this fixture only `training` had ever exercised the gate. Keras
# reserves `mask` on `call` exactly as it reserves `training` and supplies it from
# `Layer.__call__`, so an input signature covering only `x` raises on the first call
# and must be withheld; the bare decorator runs.
class Masked(tf.keras.Model):
    def __init__(self):
        super(Masked, self).__init__()
        self.conv = tf.keras.layers.Conv2D(4, 3, padding="same")

    def call(self, x, mask=None):
        return self.conv(x)


model = Masked()
image_data = tf.ones((2, 8, 8, 3))
y = model(image_data)
assert y.shape == (2, 8, 8, 4)
