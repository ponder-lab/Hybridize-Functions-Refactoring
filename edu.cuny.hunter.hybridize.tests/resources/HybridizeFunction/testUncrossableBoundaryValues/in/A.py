# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1075

import tensorflow as tf

v = tf.Variable([1.0, 2.0])


def returns_variable(x):
    return v


def returns_variable_pair(x):
    return [(v, x)]


def returns_variable_pairs(x):
    m = []
    m += [(v, x)]
    return m


def returns_tensor_array(x):
    ta = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    return x, {"parent_ids": ta}


def selects_keys(dictionary):
    selected = {}
    for key, value in dictionary.items():
        if key.startswith("a_"):
            selected[key[2:]] = value
    return selected


class Weights(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=[input_shape[-1], 2])

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

    def map_weights(self, weights):
        return [(self.kernel, weights)]


def reads_variable(x):
    return x + v


def stacks_tensor_array(x):
    ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    ta = ta.write(0, x)
    return ta.stack()


def selects_tensor_keys(dictionary):
    selected = {}
    for key, value in dictionary.items():
        if key.startswith("a_"):
            selected[key[2:]] = value
    return selected


t = tf.constant([1.0, 2.0])
returns_variable(t)
returns_variable_pair(t)
returns_variable_pairs(t)
returns_tensor_array(t)
selects_keys({"a_x": t, "b_y": t})
selects_keys(
    {"a_x": tf.TensorSpec(t.shape, t.dtype), "b_y": tf.TensorSpec(t.shape, t.dtype)}
)
layer = Weights()
layer(tf.ones([1, 3]))
layer.map_weights(tf.ones([3, 2]))
reads_variable(t)
stacks_tensor_array(t)
selects_tensor_keys({"a_x": t, "b_y": t})
