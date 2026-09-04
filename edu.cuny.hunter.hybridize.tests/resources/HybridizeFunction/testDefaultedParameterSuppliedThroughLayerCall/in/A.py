import tensorflow as tf


class Index(tf.keras.layers.Layer):
    # `k` is defaulted and not a tensor, so it carries no spec and can only be left out of a
    # signature if nothing supplies it. Its supplier calls the LAYER rather than naming `call`,
    # so a scan for calls to `call` finds none and the parameter reads as never supplied. The
    # signature then covers one argument, the dispatch passes two, and the call is rejected on
    # arity before any argument is examined.
    def call(self, queries, k=None):
        k = k if k is not None else 2
        return tf.reduce_sum(queries) * tf.cast(k, tf.float32)


class Caller(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self._index = Index()

    def call(self, queries):
        return self._index(queries, k=3)


class Named(tf.keras.layers.Layer):
    # Control: the same shape of parameter, supplied by NAMING the method rather than through the
    # dispatch. This one the call-site scan can already see, so it must stay blocked whatever
    # happens to the case above; if it ever stops being blocked the fix widened too far.
    def call(self, queries, k=None):
        k = k if k is not None else 2
        return tf.reduce_sum(queries) * tf.cast(k, tf.float32)


class NamedCaller(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self._named = Named()

    def call(self, queries):
        return self._named.call(queries, k=3)


class Unsupplied(tf.keras.layers.Layer):
    # Control: nothing supplies `k` at all, so leaving it out is sound and a signature should
    # still be emitted. This is what a fix must not break.
    def call(self, queries, k=None):
        k = k if k is not None else 2
        return tf.reduce_sum(queries) * tf.cast(k, tf.float32)


class UnsuppliedCaller(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self._plain = Unsupplied()

    def call(self, queries):
        return self._plain(queries)


class Base(tf.keras.Model):
    pass


class Injected(Base):
    # The subject's shape: the callee reaches its caller as a CONSTRUCTOR argument rather than
    # being built inline, and its Keras base is reached through an intermediate class. If the
    # receiver does not resolve, the supplying call is attributed to nothing and the parameter
    # reads as unsupplied even though the dispatch passes it.
    def call(self, queries, k=None):
        k = k if k is not None else 2
        return tf.reduce_sum(queries) * tf.cast(k, tf.float32)


class InjectedCaller(tf.keras.layers.Layer):
    def __init__(self, index):
        super().__init__()
        self._index = index

    def call(self, queries):
        return self._index(queries, k=3)


InjectedCaller(Injected())(tf.zeros((2, 4)))

Caller()(tf.zeros((2, 4)))
NamedCaller()(tf.zeros((2, 4)))
UnsuppliedCaller()(tf.zeros((2, 4)))
