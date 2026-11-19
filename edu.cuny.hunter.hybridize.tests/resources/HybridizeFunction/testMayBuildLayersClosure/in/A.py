# A function defined in another function, calling a layer it reads from a variable of the enclosing function.

import tensorflow as tf


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(2)
        self.blocks = [tf.keras.layers.Dense(2), tf.keras.layers.Dense(2)]

    def call(self, x):
        return self.dense(x)

    def predict_one(self, x):
        return self(x)


def make_first():
    return Net()


def make_second():
    return Net()


def run(body, x):
    # Calls `body` twice, with inputs of two shapes, so a decorated function it calls is traced twice.
    return body(x), body(tf.concat([x, x], axis=0))


def sample(model, x):
    # Each call defines its own `step`, which reads the one model the call is passed.
    def step(y):
        return model.predict_one(y)

    return run(step, x)


def sample_direct(model, x):
    def step_direct(y):
        return model(y)

    return run(step_direct, x)


def sample_field(model, x):
    def step_field(y):
        return model.dense(y)

    return run(step_field, x)


def sample_local(make, x):
    # Bound once, by the one assignment.
    model = make()

    def step_local(y):
        return model(y)

    return run(step_local, x)


def sample_rebound(model, other, x):
    def step_rebound(y):
        return model(y)

    first = step_rebound(x)
    # The second call of `step_rebound` reads the other model.
    model = other
    return first, step_rebound(tf.concat([x, x], axis=0))


def sample_nonlocal(model, other, x):
    def swap():
        nonlocal model
        model = other

    def step_nonlocal(y):
        return model(y)

    first = step_nonlocal(x)
    swap()
    return first, step_nonlocal(tf.concat([x, x], axis=0))


def sample_loop(models, x):
    def step_loop(y):
        return model(y)

    outs = []

    # One assignment, which runs once per model.
    for model in models:
        outs.append(step_loop(x))
        x = tf.concat([x, x], axis=0)

    return outs


def sample_nonlocal_once(model, other, x):
    def swap_once():
        nonlocal model
        model = other

    def step_nonlocal_once(y):
        return model(y)

    first = step_nonlocal_once(x)
    swap_once()
    return first, step_nonlocal_once(tf.concat([x, x], axis=0))


def sample_while(first, second, x):
    def step_while(y):
        return model(y)

    outs = []
    i = 0

    # One assignment, which runs once per iteration.
    while i < 2:
        model = first if i == 0 else second
        outs.append(step_while(x))
        x = tf.concat([x, x], axis=0)
        i += 1

    return outs


def sample_either(first, second, flag, x):
    # Bound once, to either model.
    model = first if flag else second

    def step_either(y):
        return model(y)

    return run(step_either, x)


def sample_field_rebound(model, other, x):
    def step_field_rebound(y):
        return model.dense(y)

    first = step_field_rebound(x)
    # The second call of `step_field_rebound` reads the other model's layer.
    model.dense = other.dense
    return first, step_field_rebound(tf.concat([x, x], axis=0))


def sample_indexed(model, x):
    index = [0]

    def step_indexed(y):
        # The index chooses the layer, and differs between the calls.
        return model.blocks[index[0]](y)

    first = step_indexed(x)
    index[0] = 1
    return first, step_indexed(tf.concat([x, x], axis=0))


def sample_for_parameter(model, models, x):
    def step_for_parameter(y):
        return model(y)

    outs = []

    # The loop binds the parameter again on each iteration.
    for model in models:
        outs.append(step_for_parameter(x))
        x = tf.concat([x, x], axis=0)

    return outs


def sample_for_then_assigned(models, make, x):
    def step_for_then_assigned(y):
        return model(y)

    outs = []

    for model in models:
        outs.append(step_for_then_assigned(x))
        x = tf.concat([x, x], axis=0)

    # The loop's variable, assigned once more after the loop.
    model = make()
    return outs, model


def sample_unpacked(model, other, x):
    def step_unpacked(y):
        return model(y)

    first = step_unpacked(x)
    # The second call of `step_unpacked` reads the other model, which an unpacking assignment binds.
    model, _ = other, None
    return first, step_unpacked(tf.concat([x, x], axis=0))


class Holder:

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        return self.model

    def __exit__(self, *args):
        return False


def sample_with(model, other, x):
    def step_with(y):
        return model(y)

    first = step_with(x)

    # The second call of `step_with` reads the other model, which the `with` statement binds.
    with Holder(other) as model:
        second = step_with(tf.concat([x, x], axis=0))

    return first, second


class Stack(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.blocks = [tf.keras.layers.Dense(2), tf.keras.layers.Dense(2)]

    def call(self, x, past=None):
        # As many blocks run as `past` has entries along its second axis.
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * 2
        outs = []

        for i, (block, layer_past) in enumerate(zip(self.blocks, pasts)):
            x = block(x)
            outs.append(x)

        return tf.stack(outs, axis=1)


class Wrapped(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.model = Stack()

    def call(self, inputs, past=None):
        return self.model(inputs, past)

    def predict_past(self, inputs, past=None):
        return self(inputs, past)


def sample_past(model, x):
    def step_past(tokens, past=None):
        return model.predict_past(tokens, past)

    # The second call passes a tensor where the first passed `None`, so a decorated `step_past` is traced twice, running both blocks.
    past = step_past(x)
    return step_past(x, past)


def sample_past_short(model, x):
    def step_past_short(tokens, past=None):
        return model.predict_past(tokens, past)

    # The first call runs one block and the second both, so a decorated `step_past_short` first calls the second block on a retrace.
    first = step_past_short(x, tf.ones([1, 1, 2]))
    return first, step_past_short(x)


def apply_inner(layer, x):
    # One function for every call, so the layer `inner` reads differs between the calls.
    def inner(y):
        return layer(y)

    return inner(x)


x = tf.ones([1, 3])

# Either model, as far as the analysis can tell.
shared = make_first() if x.shape[0] == 1 else make_second()


def step_global(y):
    # One function for the program, reading a variable of the script, which `swap_global` binds again.
    return shared(y)


def swap_global():
    global shared
    shared = make_second()


step_global(x)
swap_global()
step_global(tf.concat([x, x], axis=0))

held = make_first() if x.shape[0] == 1 else make_second()


def step_global_field(y):
    # The same, reading a layer from a field of the script's variable.
    return held.dense(y)


def swap_held():
    global held
    held = make_second()


step_global_field(x)
swap_held()
step_global_field(tf.concat([x, x], axis=0))

sample(make_first(), x)
sample(make_second(), x)

sample_direct(make_first(), x)
sample_direct(make_second(), x)

sample_field(make_first(), x)
sample_field(make_second(), x)

sample_local(make_first, x)
sample_local(make_second, x)

sample_rebound(make_first(), make_second(), x)
first = make_first()
second = make_second()
sample_nonlocal(first, second, x)
sample_nonlocal(second, first, x)
sample_nonlocal_once(make_first(), make_second(), x)
sample_while(make_first(), make_second(), x)
sample_either(make_first(), make_second(), True, x)
sample_either(make_first(), make_second(), False, x)
sample_field_rebound(make_first(), make_second(), x)
sample_loop([make_first(), make_second()], x)
sample_indexed(make_first(), x)
sample_indexed(make_second(), x)
sample_for_parameter(make_first(), [make_first(), make_second()], x)
sample_for_then_assigned([make_first(), make_second()], make_first, x)
sample_unpacked(make_first(), make_second(), x)
sample_with(make_first(), make_second(), x)
sample_past(Wrapped(), x)
sample_past_short(Wrapped(), x)

apply_inner(make_first(), x)
apply_inner(make_second(), tf.concat([x, x], axis=0))
