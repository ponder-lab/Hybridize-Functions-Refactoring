# Fixture for #1323: the base method is called through `super()`, unbound through the class,
# and bound, each passing the default positionally.
import tensorflow as tf


class Base:

    def f(self, x, axis=-1):
        return tf.reduce_sum(x, axis=axis)


class Derived(Base):

    def g(self, x):
        return super().f(x, -1)


d = Derived()
d.g(tf.constant([1.0, 2.0]))
Base.f(d, tf.constant([3.0, 4.0]), -1)
d.f(tf.constant([5.0, 6.0]), -1)
