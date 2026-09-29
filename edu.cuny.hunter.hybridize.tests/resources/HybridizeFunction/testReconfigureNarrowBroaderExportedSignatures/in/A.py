# A broader supplied signature on a function the program exports as a SavedModel signature (#808). The saved object is a different one,
# so only the `signatures` argument exports the method. The exported interface fixes the signature for its consumers as well as for the
# reachable callers, so the narrowing is declined.
import tensorflow as tf


class M(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def f(self, t):
        return t + 1


if __name__ == "__main__":
    m = M()
    m.f(tf.constant(2.0))
    tf.saved_model.save(tf.Module(), "exported", signatures=m.f)
