import numpy as np
import tensorflow as tf


@tf.function
def opaque(x):
    # Computes through an op the analysis can't resolve: the callee is fetched with `getattr`.
    return getattr(tf, "reduce_sum")(x)


@tf.function
def barren(x):
    return x


def opaque_eager(x):
    return getattr(tf, "reduce_sum")(x)


# Barren hybrid functions whose only calls are Python builtins, or methods of builtin values, which Ariadne leaves without a target.


@tf.function
def builtin_function(x):
    max(1, 2)
    return x


@tf.function
def string_method(x):
    "{}".format(1)
    return x


@tf.function
def dict_method(x):
    d = {"a": 1}
    d.get("a")
    return x


@tf.function
def list_method(x):
    l = []
    l.append(x)
    return x


@tf.function
def dict_items_loop(x):
    d = {"a": 1}
    for k, v in d.items():
        pass
    return x


@tf.function
def tuple_method(x):
    p = (1, 2)
    p.count(1)
    return x


@tf.function
def set_method(x):
    s = {1}
    s.add(2)
    return x


@tf.function
def numpy_call(x):
    np.zeros(3)
    return x


t = tf.constant([1.0, 2.0, 3.0])
opaque(t)
barren(t)
opaque_eager(t)
builtin_function(t)
string_method(t)
dict_method(t)
list_method(t)
dict_items_loop(t)
tuple_method(t)
set_method(t)
numpy_call(t)
