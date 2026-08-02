import tensorflow as tf

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]


class Model:

    def __init__(self):
        # An attribute store reusing the constant's name with a different literal: attributes are not name
        # bindings and must not compete with the module-level constant the decorators below reference.
        self.train_step_signature = [tf.TensorSpec(shape=(None,), dtype=tf.int32)]

    @tf.function(input_signature=train_step_signature)
    def train_step(self, inputs, targets):
        return inputs + targets

    @tf.function(input_signature=train_step_signature)
    def test_step(self, inputs, targets):
        return inputs - targets


if __name__ == "__main__":
    m = Model()
    ones = tf.ones([2, 3], dtype=tf.int32)
    assert int(tf.reduce_sum(m.train_step(ones, ones))) == 12
    assert int(tf.reduce_sum(m.test_step(ones, ones))) == 0
