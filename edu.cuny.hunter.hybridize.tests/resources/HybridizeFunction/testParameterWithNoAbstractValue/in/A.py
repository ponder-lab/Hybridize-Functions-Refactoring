import tensorflow as tf


@tf.function
def p_fmt(s):
    # Its argument comes from `str.format`, a method the analysis doesn't model, so it has no abstract value.
    return s


@tf.function
def p_int(n):
    # Its argument is a literal the analysis models, so its non-tensor classification is a determination.
    return n


p_fmt("_{}".format(3))
p_int(3)
