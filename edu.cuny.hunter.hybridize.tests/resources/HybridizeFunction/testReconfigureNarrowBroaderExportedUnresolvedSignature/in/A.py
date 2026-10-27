# A broader supplied signature in a program that exports a SavedModel signature the analysis cannot resolve (#808). The signature is a
# concrete function of a loaded model, which the summaries do not model, so what the save exports is unknown, and the narrowing is
# declined.
import tensorflow as tf


class M(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def f(self, t):
        return t + 1


class N(tf.Module):
    pass


if __name__ == "__main__":
    m = M()
    m.f(tf.constant(2.0))
    loaded = tf.saved_model.load("other")
    tf.saved_model.save(
        N(),
        "exported",
        signatures={"serving_default": loaded.f.get_concrete_function()},
    )
