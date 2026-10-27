import tensorflow as tf


# A `call` override whose class is not a Keras layer or model (#1099): the bases form a
# diamond over `object`, so the engine's walk reaches `Base` twice and `object` once,
# and finds no Keras class. Nothing supplies `training`, so the parameter is left out
# of the inferred signature, which runs.
class Base(object):
    pass


class Left(Base):
    pass


class Right(Base):
    pass


class Diamond(Left, Right):
    def call(self, x, training=False):
        return tf.nn.relu(x)


y = Diamond().call(tf.random.normal([1, 3]))
assert y.shape == (1, 3)
