# Name-referenced variant of the supplied-tighter adjudication (#834, #808). The decorator references the tighter signature through a
# module-level constant. The disagreement is reported and both the reference and the constant are left unchanged (the constant may have
# other users). The supplied signature intentionally disagrees with the call sites, so this fixture is analyzed statically rather than
# executed.
import tensorflow as tf

f_signature = [tf.TensorSpec(shape=(2,), dtype=tf.float32)]


@tf.function(input_signature=f_signature)
def f(t):
    return t + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
    f(tf.ones([2]))
