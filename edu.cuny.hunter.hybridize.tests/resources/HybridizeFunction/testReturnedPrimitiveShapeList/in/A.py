# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1117

import tensorflow as tf


def shape_list(x, out_type=tf.int32):
    static = x.shape.as_list()
    dynamic = tf.shape(x, out_type=out_type)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class Projection(tf.keras.layers.Layer):
    def call(self, inputs):
        _, _, dmodel = shape_list(inputs)
        return tf.keras.layers.Dense(units=dmodel)(inputs)


x = tf.ones([2, 3, 5])
print(Projection()(x).shape)

inputs = tf.keras.Input(shape=(None, 5))
model = tf.keras.Model(inputs, Projection()(inputs))
print(model.output_shape)
