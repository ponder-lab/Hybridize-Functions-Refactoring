import tensorflow as tf

ambiguous_signature = [tf.TensorSpec(shape=(None,), dtype=tf.float32)]
ambiguous_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.float32)]


def make_signature():
    return [tf.TensorSpec(shape=(None,), dtype=tf.float32)]


computed_signature = make_signature()

shadowed_signature = [tf.TensorSpec(shape=(None,), dtype=tf.float32)]

for loop_signature in [[tf.TensorSpec(shape=(None,), dtype=tf.float32)]]:
    pass


@tf.function(input_signature=ambiguous_signature)
def reassigned(t):
    return t + 1


@tf.function(input_signature=computed_signature)
def computed(t):
    return t + 1


# The sole binding is a loop target, not a module-level assignment; the resolution only models plain
# single-target assignments and must decline.
@tf.function(input_signature=loop_signature)
def loop_bound(t):
    return t + 1


class Shadowing:
    # The class-body binding is what the decorator below actually sees at decoration time; the sole-binding
    # rule declines rather than wrongly resolving to the module-level literal above.
    shadowed_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.float32)]

    @tf.function(input_signature=shadowed_signature)
    def shadowed(self, t):
        return t + 1


if __name__ == "__main__":
    v = tf.ones([3])
    m = tf.ones([2, 2])
    assert (
        float(tf.reduce_sum(reassigned(m))) == 8.0
    )  # binds to the second (rank-2) assignment
    assert float(tf.reduce_sum(computed(v))) == 6.0
    assert float(tf.reduce_sum(loop_bound(v))) == 6.0
    assert (
        float(tf.reduce_sum(Shadowing().shadowed(m))) == 8.0
    )  # binds to the class-body (rank-2) literal
