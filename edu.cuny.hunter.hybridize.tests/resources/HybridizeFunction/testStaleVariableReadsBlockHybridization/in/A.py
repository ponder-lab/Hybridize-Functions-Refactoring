import tensorflow as tf


class Tiny(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d = tf.keras.layers.Dense(2)

    def call(self, x):
        return self.d(x)


model = Tiny()
optimizer = tf.keras.optimizers.Adam()


def stale_read(x):
    tv = model.trainable_variables
    with tf.GradientTape() as g:
        loss = tf.reduce_sum(model(x))
    grads = g.gradient(loss, tv)
    optimizer.apply_gradients(zip(grads, tv))
    return loss


def fresh_read(x):
    with tf.GradientTape() as g:
        loss = tf.reduce_sum(model(x))
    grads = g.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


a = tf.ones((2, 3))

# Eagerly the stale snapshot is only silently empty on the first call; the raise
# needs tracing (the variable-lifting re-trace), which is what the precondition
# prevents the refactoring from introducing.
assert stale_read(a).shape == ()
assert stale_read(a).shape == ()
assert fresh_read(a).shape == ()
