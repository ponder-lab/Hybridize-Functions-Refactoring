# A broader supplied signature on a method of an object saved through an imported alias of the SavedModel module, from inside a function
# (#808). The alias names the same export, so the narrowing is declined.
from tensorflow import saved_model
import tensorflow as tf


class M(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def f(self, t):
        return t + 1


def export(model):
    saved_model.save(model, "exported")


if __name__ == "__main__":
    m = M()
    m.f(tf.constant(2.0))
    export(m)
