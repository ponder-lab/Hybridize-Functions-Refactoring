# A broader supplied signature on a function whose concrete function enters a SavedModel `signatures` dictionary built by the caller of
# the function that saves another object (#808). That function adds an entry of its own, but the dictionary is not a literal it
# allocates, so its other entries are seen only through its fields. The concrete function's entry points to nothing the analysis can
# see, so what the save exports is unknown, and the narrowing is declined.
import tensorflow as tf


class M(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def f(self, t):
        return t + 1

    @tf.function
    def g(self, t):
        return t


class N(tf.Module):
    pass


def export(model, signatures):
    signatures["identity"] = model.g
    tf.saved_model.save(N(), "exported", signatures=signatures)


if __name__ == "__main__":
    m = M()
    m.f(tf.constant(2.0))
    export(m, {"serving_default": m.f.get_concrete_function()})
