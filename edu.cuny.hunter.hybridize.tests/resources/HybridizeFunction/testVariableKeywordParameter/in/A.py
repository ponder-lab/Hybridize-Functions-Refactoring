import tensorflow as tf

# Pins the `**kwargs` decline of #902. An `input_signature` fixes the arguments a function
# accepts, so a rest-keyword slot stops absorbing anything and a caller passing a keyword
# raises where it used to succeed. Keras is that caller for a layer: it reads `call`'s
# declaration, sees that `**kwargs` can take `training`, and passes it, so writing a
# signature onto `Absorbing.call` would break a layer that runs today. `Positional.call`
# declares no such slot and keeps its signature, which localizes the decline.


class Absorbing(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        return tf.matmul(inputs, inputs)


class Positional(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.matmul(inputs, inputs)


def absorbing_function(x, **kwargs):
    return tf.matmul(x, x)


# Already hybrid, and already broken by its own hand: the supplied signature has disabled the
# rest-keyword slot, so a caller passing a keyword raises before this refactoring touches
# anything. Nothing can be written here, so the report is the action.
@tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
def supplied_and_absorbing(x, **kwargs):
    return tf.matmul(x, x)


Absorbing()(tf.ones((2, 2)))
Positional()(tf.ones((2, 2)))
absorbing_function(tf.ones((2, 2)))
supplied_and_absorbing(tf.ones((2, 2)))
