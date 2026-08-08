import tensorflow as tf

# Reduces TensorFlow2.0-Examples' YOLOv3 (#887): a function factored out of Keras
# Functional model construction receives the symbolic input. `tf.function` is one of the
# APIs a `KerasTensor` refuses, so decorating `symbolic` (or `derived`, whose argument is
# the symbolic input threaded through a built-in layer) raises a `TypeError` on the first
# call, before anything is traced. `eager`, called only with a real tensor, is unaffected.


def symbolic(input_layer):
    return tf.abs(input_layer)


def derived(feature):
    return tf.abs(feature)


def eager(x):
    return tf.abs(x)


input_tensor = tf.keras.layers.Input([4])
symbolic_output = symbolic(input_tensor)
derived_output = derived(tf.keras.layers.Dense(3)(input_tensor))

model = tf.keras.Model(input_tensor, symbolic_output)
assert model(tf.ones((2, 4))).shape == (2, 4)

assert eager(tf.ones((2, 4))).shape == (2, 4)
