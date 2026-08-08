import tensorflow as tf


# Reduces deep_recommenders' CIN.call (#888): `cin` receives a two-tensor tuple at one call
# site and a bare tensor at another, so its nesting varies across call sites. Ariadne types
# the parameter from the unwrapped site, and a single `TensorSpec` reduced from that site
# admits none of the callers that pass the tuple. Both sites are legal, and the body indexes
# rather than unpacks, so the arms isolate the signature question from the symbolic-iteration
# hazard of #830; in the subject the unwrapped site is a negative test the body rejects, a
# separable concern. `tuples_only` receives tuples alone and keeps its nested spec.
def cin(inputs):
    return tf.matmul(inputs[0], inputs[1])


def tuples_only(inputs):
    return tf.matmul(inputs[0], inputs[1])


assert cin((tf.ones((2, 3, 5)), tf.ones((2, 5, 3)))).shape == (2, 3, 3)
assert cin(tf.ones((2, 3, 3))).shape == (3, 3)

assert tuples_only((tf.ones((2, 3, 5)), tf.ones((2, 5, 3)))).shape == (2, 3, 3)
