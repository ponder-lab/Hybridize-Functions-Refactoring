# A broader supplied signature on a function whose concrete function enters a SavedModel `signatures` dictionary built in another function
# (#808). The dictionary's entries point to nothing the analysis can see, so what the save exports is unknown, and the narrowing is
# declined.
import tensorflow as tf


class M(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def f(self, t):
        return t + 1


class N(tf.Module):
    pass


def signatures_of(model):
    return {"serving_default": model.f.get_concrete_function()}


if __name__ == "__main__":
    m = M()
    m.f(tf.constant(2.0))
    tf.saved_model.save(N(), "exported", signatures=signatures_of(m))
