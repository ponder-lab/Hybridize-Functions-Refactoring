import tensorflow as tf


# Reduces deep_recommenders' FM linear term (#883 Case 2): the parameter flows into a
# built-in Dense, whose `build` reads the last input dimension. `FM.call` is invoked at
# differing ranks, so its inferred spec is `shape=None`, and `TensorShape(None)` breaks
# the layer before the body runs; `Ranked.call`'s single rank keeps the spec writable
# (Dense admits a known rank with a wildcard axis).
class FM(tf.keras.Model):
    def __init__(self):
        super(FM, self).__init__()
        self._linear = tf.keras.layers.Dense(1)

    def call(self, sparse_inputs):
        return self._linear(sparse_inputs)


class Ranked(tf.keras.Model):
    def __init__(self):
        super(Ranked, self).__init__()
        self._linear = tf.keras.layers.Dense(1)

    def call(self, sparse_inputs):
        return self._linear(sparse_inputs)


# An unknown-rank spec into ordinary modeled ops stays emittable: only a built-in layer
# application is rank-sensitive, which this arm pins in the allowing direction.
def op_only(x):
    return tf.abs(x)


model = FM()
assert model(tf.ones((2, 5))).shape == (2, 1)
assert model(tf.ones((2, 3, 5))).shape == (2, 3, 1)

ranked = Ranked()
assert ranked(tf.ones((2, 5))).shape == (2, 1)
assert ranked(tf.ones((4, 5))).shape == (4, 1)

assert op_only(tf.ones((3,))).shape == (3,)
assert op_only(tf.ones((2, 3))).shape == (2, 3)
