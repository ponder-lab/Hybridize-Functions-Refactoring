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


def returns_dot_op(x):
    # The issue's own minimal reproduction: reading `op` on what an assignment returns yields the
    # Operation that performs it. The receiver is a plain variable rather than a TensorFlow-rooted
    # expression, so a rule that reaches the module only by walking attribute links misses this.
    return v.assign_add(tf.reduce_sum(x)).op


def returns_tensor(x):
    # Control: same shape of body, but the value returned is a Tensor, so hybridizing
    # it is sound. If a decline widened from the return type to the assignments, this
    # would be caught here.
    v.assign_add(tf.reduce_sum(x))
    return tf.reduce_sum(x) * 2.0


returns_operation(tf.zeros((4,)))
returns_tensor(tf.zeros((4,)))
returns_dot_op(tf.zeros((4,)))


def mixed_returns(x):
    # One path yields None, which the tracer accepts, and the other an Operation. The paths
    # disagree, so this is not a function every return of which is an Operation, and the
    # all-not-any rule must not decline it on the strength of the valued path alone.
    if tf.reduce_sum(x) > 0:
        return
    return tf.group([v.assign_add(1.0)])


mixed_returns(tf.zeros((4,)))


class Collector:
    def group(self, xs):
        return xs

    @property
    def op(self):
        return 0


collector = Collector()


def unrooted_names(x):
    # The sole return is a `group` call that is not TensorFlow's. Matching on the trailing name
    # alone declines this, refusing a conversion that is sound, so the check must be rooted. The
    # return has to be the only one: a second, tensor-valued return would make the all-rule fail
    # for an unrelated reason and the case would not isolate the rooting.
    return collector.group([1, 2])


def encloses_a_returner(x):
    # This function has no return of its own, so it yields None, which the tracer accepts. The
    # nested definition returns an Operation. A walk that descends into inner definitions sees
    # only that return, concludes every return is an Operation, and declines this function for a
    # return it does not make.
    def inner():
        return tf.group([v.assign_add(1.0)])

    inner()


unrooted_names(tf.zeros((4,)))
encloses_a_returner(tf.zeros((4,)))
