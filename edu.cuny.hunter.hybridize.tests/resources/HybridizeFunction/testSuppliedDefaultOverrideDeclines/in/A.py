# Fixture for #1323: the call may dispatch to either `Base.f` or `Derived.f`, whose defaults
# differ, so deleting `axis=-1` would change what `Derived.f` receives.
import tensorflow as tf


class Base:

    def f(self, x, axis=-1):
        return tf.reduce_sum(x, axis=axis)


class Derived(Base):

    def f(self, x, axis=0):
        return tf.reduce_sum(x, axis=axis)


for o in [Base(), Derived()]:
    o.f(tf.constant([1.0, 2.0]), axis=-1)
