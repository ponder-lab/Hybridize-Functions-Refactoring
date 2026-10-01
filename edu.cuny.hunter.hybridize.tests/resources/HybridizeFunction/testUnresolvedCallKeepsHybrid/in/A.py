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


# Calls that read like builtins but aren't, or that call back into a function: each may compute tensors.


def make(len):
    @tf.function
    def step(x):
        # `len` is the enclosing function's parameter, not the builtin.
        return len(x)

    return step


sum = getattr(tf, "reduce_sum")


@tf.function
def rebound(x):
    # The module rebinds `sum`.
    return sum(x)


@tf.function
def mapped(x):
    return list(map(tf.nn.relu, [x, x]))


@tf.function
def keyed(x):
    return max([x, x], key=tf.reduce_sum)


@tf.function
def sorted_in_place(x):
    l = [x, x]
    l.sort(key=tf.reduce_sum)
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
make(getattr(tf, "reduce_sum"))(t)
rebound(t)
mapped(t)
keyed(t)
sorted_in_place(t)
