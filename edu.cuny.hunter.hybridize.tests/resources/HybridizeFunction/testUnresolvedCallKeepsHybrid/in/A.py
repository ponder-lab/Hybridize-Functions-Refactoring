import os

import numpy as np
import tensorflow as tf

# An op name the analysis can't model: read from the environment, so `getattr(tf, OP)` stays unresolved however `getattr` on a
# constant name is modeled.
OP = os.environ.get("HFR_OP", "reduce_sum")


@tf.function
def opaque(x):
    # Computes through an op the analysis can't resolve: the callee is fetched with `getattr` by a name it can't model.
    return getattr(tf, OP)(x)


@tf.function
def opaque_constant(x):
    # The same through a constant name. This documents current behavior: Ariadne doesn't yet resolve `getattr` on a constant name, so
    # this call is unresolved, and a release that does (ponder-lab/ML#909) will turn it into a resolved tensor op.
    return getattr(tf, "reduce_sum")(x)


@tf.function
def barren(x):
    return x


def opaque_eager(x):
    return getattr(tf, OP)(x)


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


sum = getattr(tf, OP)


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


def setup():
    global len
    len = getattr(tf, OP)


setup()


@tf.function
def global_rebound(x):
    # `setup` rebinds `len` at module scope through a `global` declaration.
    return len(x)


@tf.function
def sorted_keyed(x):
    return sorted([x, x], key=tf.reduce_sum)


def abs(a):
    return a


@tf.function
def uses_user_abs(x):
    # `abs` is the module's own barren function, resolved as such, not the builtin.
    abs(1)
    return x


def countdown(n):
    return countdown(n - 1) if n > 0 else 0


@tf.function
def calls_recursive(x):
    # Reaches the recursive `countdown`, whose call graph has a cycle.
    countdown(3)
    return x


@tf.function
def global_builtin(x):
    # `global` makes the read of `max` a global one; with no module binding, it is the builtin.
    global max
    max(1, 2)
    return x


@tf.function
def global_keyed(x):
    global sorted
    return sorted([x, x], key=tf.reduce_sum)


class Keyed:
    def pick(self, key=None):
        return key


@tf.function
def user_keyed(x):
    # A resolved method of a user object given `key=`: not a builtin, so it is scanned like any other call.
    Keyed().pick(key=1)
    return x


t = tf.constant([1.0, 2.0, 3.0])
opaque(t)
opaque_constant(t)
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
make(getattr(tf, OP))(t)
rebound(t)
mapped(t)
keyed(t)
sorted_in_place(t)
global_rebound(t)
sorted_keyed(t)
uses_user_abs(t)
calls_recursive(t)
global_builtin(t)
global_keyed(t)
user_keyed(t)
