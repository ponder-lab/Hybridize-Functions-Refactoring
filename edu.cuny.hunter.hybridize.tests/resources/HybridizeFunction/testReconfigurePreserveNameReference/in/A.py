# Name-referenced variant of the supplied-tighter modify path (#834). The decorator references the tighter signature through a
# module-level constant; the overwrite must replace exactly the reference at the decorator site with the inferred literal, leaving
# the module-level constant itself intact (it may have other users). The supplied signature intentionally disagrees with the call
# sites, so this fixture is analyzed statically rather than executed.
import tensorflow as tf

f_signature = [tf.TensorSpec(shape=(2,), dtype=tf.float32)]


@tf.function(input_signature=f_signature)
def f(t):
    return t + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
    f(tf.ones([2]))
