# A function returning a tf.Operation cannot be hybridized: tf.function accepts only
# Tensors, ExtensionTypes, or None as return values, so the decorator itself raises
# regardless of any input signature (issue 929).
import tensorflow as tf

v = tf.Variable(0.0)
w = tf.Variable(0.0)


def returns_operation(x):
    # `tf.group` yields an Operation. So do `assign(...).op`, `tf.no_op()`, and the
    # summary writers; the annotation is incidental and may be absent entirely.
    return tf.group([v.assign_add(tf.reduce_sum(x)), w.assign_add(1.0)])


def returns_tensor(x):
    # Control: same shape of body, but the value returned is a Tensor, so hybridizing
    # it is sound. If a decline widened from the return type to the assignments, this
    # would be caught here.
    v.assign_add(tf.reduce_sum(x))
    return tf.reduce_sum(x) * 2.0


returns_operation(tf.zeros((4,)))
returns_tensor(tf.zeros((4,)))
