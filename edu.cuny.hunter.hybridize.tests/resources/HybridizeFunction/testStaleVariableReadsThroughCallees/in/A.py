import tensorflow as tf


class Tiny(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d = tf.keras.layers.Dense(2)

    def call(self, x):
        return self.d(x)


model = Tiny()
optimizer = tf.keras.optimizers.Adam()


def forward(x):
    return tf.reduce_sum(model(x))


def stale_read_through_helper(x):
    tv = model.trainable_variables
    with tf.GradientTape() as g:
        loss = forward(x)
    grads = g.gradient(loss, tv)
    optimizer.apply_gradients(zip(grads, tv))
    return loss


def fresh_read_after_helper(x):
    with tf.GradientTape() as g:
        loss = forward(x)
    grads = g.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


a = tf.ones((2, 3))

# Eagerly the stale snapshot is only silently empty on the first call; the raise needs
# tracing. Decorating stale_read_through_helper reproduces multigpu_training's failure
# verbatim on TF 2.9.3 (the singleton-variable ValueError in optimizer_v2.add_slot):
# the helper's forward pass builds the model inside the trace, the variable-lifting
# re-trace re-executes the snapshot populated, and slot creation lands on a non-first
# trace. The helper is the point of the fixture: the receiver is never invoked in the
# reading function's own body.
assert stale_read_through_helper(a).shape == ()
assert stale_read_through_helper(a).shape == ()
assert fresh_read_after_helper(a).shape == ()
