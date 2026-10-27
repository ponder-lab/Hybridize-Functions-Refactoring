# A broader supplied signature on a function the program exports through a SavedModel `signatures` dictionary whose entry is a concrete
# function (#808). The saved object is a different one, so only the dictionary's entry exports the method, and its narrowing is declined.
# The dictionary's entries are all resolved, so an unrelated function's narrowing still applies.
import tensorflow as tf


class M(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def f(self, t):
        return t + 1


class N(tf.Module):
    pass


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
def g(t):
    return t + 1


if __name__ == "__main__":
    m = M()
    m.f(tf.constant(2.0))
    g(tf.constant(2.0))
    tf.saved_model.save(
        N(), "exported", signatures={"serving_default": m.f.get_concrete_function()}
    )
