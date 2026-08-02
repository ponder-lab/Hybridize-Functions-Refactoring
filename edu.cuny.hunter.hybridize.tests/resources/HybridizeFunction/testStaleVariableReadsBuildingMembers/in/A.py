import tensorflow as tf


class Tiny(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d = tf.keras.layers.Dense(2)

    def call(self, x):
        return self.d(x)


model = Tiny()
model.compile(optimizer="sgd", loss="mse")
optimizer = tf.keras.optimizers.Adam()

xs = tf.ones((2, 3))
ys = tf.ones((2, 2))


def pairs():
    while True:
        yield xs, ys


def inputs_only():
    while True:
        yield xs


# Each arm reads the variable collection only AFTER a member that triggers the lazy build
# (runtime-verified on TF 2.9.3 to flip `built`), so the read is fresh and must not be
# flagged stale. Passing-precondition assertions are deferred to #836: the training-surface
# members themselves are eager-only under tracing, which is a separate, unmodeled hazard.
def fresh_after_predict(x):
    model.predict(x)
    tv = model.trainable_variables
    with tf.GradientTape() as g:
        loss = tf.reduce_sum(model(x))
    grads = g.gradient(loss, tv)
    optimizer.apply_gradients(zip(grads, tv))
    return loss


def fresh_after_evaluate(x):
    model.evaluate(x, ys, verbose=0)
    tv = model.trainable_variables
    with tf.GradientTape() as g:
        loss = tf.reduce_sum(model(x))
    grads = g.gradient(loss, tv)
    optimizer.apply_gradients(zip(grads, tv))
    return loss


def fresh_after_train_on_batch(x):
    model.train_on_batch(x, ys)
    tv = model.trainable_variables
    with tf.GradientTape() as g:
        loss = tf.reduce_sum(model(x))
    grads = g.gradient(loss, tv)
    optimizer.apply_gradients(zip(grads, tv))
    return loss


def fresh_after_test_on_batch(x):
    model.test_on_batch(x, ys)
    tv = model.trainable_variables
    with tf.GradientTape() as g:
        loss = tf.reduce_sum(model(x))
    grads = g.gradient(loss, tv)
    optimizer.apply_gradients(zip(grads, tv))
    return loss


def fresh_after_predict_on_batch(x):
    model.predict_on_batch(x)
    tv = model.trainable_variables
    with tf.GradientTape() as g:
        loss = tf.reduce_sum(model(x))
    grads = g.gradient(loss, tv)
    optimizer.apply_gradients(zip(grads, tv))
    return loss


def fresh_after_fit_generator(x):
    model.fit_generator(pairs(), steps_per_epoch=1, epochs=1, verbose=0)
    tv = model.trainable_variables
    with tf.GradientTape() as g:
        loss = tf.reduce_sum(model(x))
    grads = g.gradient(loss, tv)
    optimizer.apply_gradients(zip(grads, tv))
    return loss


def fresh_after_evaluate_generator(x):
    model.evaluate_generator(pairs(), steps=1)
    tv = model.trainable_variables
    with tf.GradientTape() as g:
        loss = tf.reduce_sum(model(x))
    grads = g.gradient(loss, tv)
    optimizer.apply_gradients(zip(grads, tv))
    return loss


def fresh_after_predict_generator(x):
    model.predict_generator(inputs_only(), steps=1)
    tv = model.trainable_variables
    with tf.GradientTape() as g:
        loss = tf.reduce_sum(model(x))
    grads = g.gradient(loss, tv)
    optimizer.apply_gradients(zip(grads, tv))
    return loss


assert fresh_after_predict(xs).shape == ()
assert fresh_after_evaluate(xs).shape == ()
assert fresh_after_train_on_batch(xs).shape == ()
assert fresh_after_test_on_batch(xs).shape == ()
assert fresh_after_predict_on_batch(xs).shape == ()
assert fresh_after_fit_generator(xs).shape == ()
assert fresh_after_evaluate_generator(xs).shape == ()
assert fresh_after_predict_generator(xs).shape == ()
