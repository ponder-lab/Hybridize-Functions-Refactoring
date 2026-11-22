# Functions defined in a function that a trace runs, each calling a layer it reads from a variable of that function.

import tensorflow as tf


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(2)

    def call(self, x):
        return self.dense(x)


def make_first():
    return Net()


def make_second():
    return Net()


class Holder:

    def __init__(self, layer):
        self.layer = layer


def apply_inner(layer, x):
    def inner(y):
        return layer(y)

    return inner(x)


def apply_bare(layer, x):
    # Passes `inner_bare` nothing.
    def inner_bare():
        return layer(x)

    return inner_bare()


def apply_alias(layer, x):
    chosen = layer

    def inner_alias(y):
        return chosen(y)

    return inner_alias(x)


def apply_held(holder, x):
    def inner_held(y):
        return holder.layer(y)

    return inner_held(x)


def apply_deep(layer, x):
    def middle(y):
        def innermost(z):
            return layer(z)

        return innermost(y)

    return middle(x)


def call_layer(layer, y):
    return layer(y)


def apply_through(layer, x):
    # `inner_through` passes the layer it reads on.
    def inner_through(y):
        return call_layer(layer, y)

    return inner_through(x)


def apply_sited(layer, x):
    # Passed two layers that the script itself creates.
    def inner_sited(y):
        return layer(y)

    return inner_sited(x)


def apply_once(layer, x):
    # Passed one layer by both of its calls.
    def inner_once(y):
        return layer(y)

    return inner_once(x)


def apply_bare_through(layer, x):
    # `inner_bare_through` is passed nothing, and passes the layer it reads on.
    def inner_bare_through():
        return call_layer(layer, x)

    return inner_bare_through()


def apply_same(layer, x):
    # Passed one tensor by both of its calls, so `inner_same` is passed nothing that differs between them.
    def inner_same(y):
        return call_layer(layer, y)

    return inner_same(x)


def apply_lambda(layer, x):
    # The function reading the parameter is a lambda expression.
    inner_lambda = lambda y: layer(y)

    return inner_lambda(x)


def apply_reexported(layer, x):
    # `middle_reexported` binds a variable of its own to the layer it reads.
    def middle_reexported(y):
        chosen = layer

        def innermost_reexported(z):
            return chosen(z)

        return innermost_reexported(y)

    return middle_reexported(x)


def make_caller(layer):
    def caller(y):
        return layer(y)

    return caller


def use_made(x):
    # Calls a function that a call outside any trace of it defined.
    return call_made(x)


class Runner:
    # No layer, and no model.

    def __init__(self, net):
        self.net = net

    def run_field(self, x):
        layer = self.net

        def inner_field(y):
            return layer(y)

        return inner_field(x)

    def run_self(self, x):
        def inner_self(y):
            return self.net(y)

        return inner_self(x)

    def run_passed(self, layer, x):
        def inner_passed(y):
            return layer(y)

        return inner_passed(x)

    def run_passed_once(self, layer, x):
        def inner_passed_once(y):
            return layer(y)

        return inner_passed_once(x)

    def run_direct_once(self, layer, x):
        return layer(x)


x = tf.ones([1, 3])
wide = tf.concat([x, x], axis=0)

apply_inner(make_first(), x)
apply_inner(make_second(), wide)

apply_bare(make_first(), x)
apply_bare(make_second(), wide)

apply_alias(make_first(), x)
apply_alias(make_second(), wide)

apply_held(Holder(make_first()), x)
apply_held(Holder(make_second()), wide)

apply_deep(make_first(), x)
apply_deep(make_second(), wide)

apply_through(make_first(), x)
apply_through(make_second(), wide)

apply_sited(Net(), x)
apply_sited(Net(), wide)

once = make_first()
apply_once(once, x)
apply_once(once, wide)

apply_bare_through(make_first(), x)
apply_bare_through(make_second(), wide)

apply_same(make_first(), x)
apply_same(make_second(), x)

apply_lambda(make_first(), x)
apply_lambda(make_second(), wide)

apply_reexported(make_first(), x)
apply_reexported(make_second(), wide)

call_made = make_caller(make_first())
use_made(x)
use_made(wide)

field_first = Runner(make_first())
field_second = Runner(make_second())
field_first.run_field(x)
field_first.run_field(wide)
field_second.run_field(x)
field_second.run_field(wide)

self_first = Runner(make_first())
self_second = Runner(make_second())
self_first.run_self(x)
self_first.run_self(wide)
self_second.run_self(x)
self_second.run_self(wide)

passing = Runner(make_first())
passing.run_passed(make_first(), x)
passing.run_passed(make_second(), wide)

passing_once = Runner(make_first())
passed_once = make_first()
passing_once.run_passed_once(passed_once, x)
passing_once.run_passed_once(passed_once, wide)

direct_once = Runner(make_first())
called_once = make_first()
direct_once.run_direct_once(called_once, x)
direct_once.run_direct_once(called_once, wide)
