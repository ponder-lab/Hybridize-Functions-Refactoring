import tensorflow as tf

# Reduces TensorFlow2.0-Examples' YOLOv3 (#1097): the `KerasTensor` reaches a call site
# from another frame. `model_fn` takes the `Input` directly, as in #887. `upsample` takes
# a built-in layer's output over `model_fn`'s parameter, and `head` takes `model_fn`'s
# return value; both receive a `KerasTensor` at run time, so decorating either raises a
# `TypeError` on the first call. The `_eager` arms repeat the same shapes rooted at a real
# tensor instead, which the Functional API leaves eager, so they stay convertible.
# `scale_inside` is called from a layer's own `call`, which Keras gives a placeholder
# rather than the `KerasTensor` the layer is applied to, so it stays convertible too.


def upsample(input_layer):
    return tf.image.resize(
        input_layer,
        (input_layer.shape[1] * 2, input_layer.shape[2] * 2),
        method="nearest",
    )


def model_fn(input_layer):
    conv = tf.keras.layers.Conv2D(8, 3, padding="same")(input_layer)
    return upsample(conv)


def head(t):
    return t * 2.0


def upsample_eager(input_layer):
    return tf.image.resize(
        input_layer,
        (input_layer.shape[1] * 2, input_layer.shape[2] * 2),
        method="nearest",
    )


def model_fn_eager(input_layer):
    conv = tf.keras.layers.Conv2D(8, 3, padding="same")(input_layer)
    return upsample_eager(conv)


def head_eager(t):
    return t * 2.0


def scale_inside(x):
    return x * 2.0


class Scale(tf.keras.layers.Layer):
    def call(self, inputs):
        return scale_inside(inputs)


inp = tf.keras.layers.Input([16, 16, 3])
out = model_fn(inp)
y = head(out)
scaled = Scale()(inp)

out_eager = model_fn_eager(tf.ones([1, 16, 16, 3]))
y_eager = head_eager(out_eager)
assert y_eager.shape == (1, 32, 32, 8)
