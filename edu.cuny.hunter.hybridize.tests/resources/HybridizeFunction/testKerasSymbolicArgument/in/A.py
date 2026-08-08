import tensorflow as tf

# Reduces TensorFlow2.0-Examples' YOLOv3 (#887): a function factored out of Keras
# Functional model construction receives the symbolic input. `tf.function` is one of the
# APIs a `KerasTensor` refuses, so decorating `symbolic` (whose argument is the symbolic
# input), `derived` (the same input threaded through a built-in layer), or `by_keyword`
# (the same input passed by keyword) raises a `TypeError` on the first call, before
# anything is traced. `eager` is called only with a real tensor, and `merged` receives the
# symbolic input on one path and a real tensor on the other, which leaves its argument's
# symbolicness path-dependent; both stay convertible.


def symbolic(input_layer):
    return tf.abs(input_layer)


def derived(feature):
    return tf.abs(feature)


def by_keyword(feature):
    return tf.abs(feature)


def eager(x):
    return tf.abs(x)


def merged(x):
    return tf.abs(x)


input_tensor = tf.keras.layers.Input([4])
symbolic_output = symbolic(input_tensor)
derived_output = derived(tf.keras.layers.Dense(3)(input_tensor))
keyword_output = by_keyword(feature=input_tensor)

model = tf.keras.Model(input_tensor, symbolic_output)
assert model(tf.ones((2, 4))).shape == (2, 4)

assert eager(tf.ones((2, 4))).shape == (2, 4)

if tf.executing_eagerly():
    pick = tf.ones((2, 4))
else:
    pick = input_tensor

assert merged(pick).shape == (2, 4)
