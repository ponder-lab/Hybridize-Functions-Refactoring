# Singleton-non-concrete coverage: the parameter receives a tensor whose dtype Ariadne resolves
# (float32, the `tf.keras.Input` default) but whose shape it cannot trace through `json.loads`.
# `inferSpec`'s `single.getDims() == null` branch fires and `inferInputSignature` returns
# `Optional.empty`.

import json

import tensorflow as tf


def func(t):
    return t


shape = json.loads("[32]")
x = tf.keras.Input(shape=shape)
assert x.dtype == tf.float32
func(x)
