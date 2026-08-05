import tensorflow as tf


# A user class defining its own `predict`: the same member name as the guarded Keras
# endpoint, but dispatching to user code (the in-corpus override shape). Calling it must
# not block by name.
class Estimator:
    def predict(self, t):
        return tf.reduce_sum(t)


def calls_override(e, t):
    return e.predict(t)


# Calls the framework's own guarded endpoint on a Keras model: raises RuntimeError inside
# a tf.function, so hybridizing this function converts working code into raising code.
def calls_fit(m, x, y):
    m.fit(x, y, epochs=1, verbose=0)
    return tf.reduce_sum(x)


inp = tf.keras.Input(shape=(3,))
out = tf.keras.layers.Dense(2)(inp)
model = tf.keras.Model(inp, out)
model.compile(optimizer="sgd", loss="mse")

est = Estimator()
xs = tf.ones((4, 3))
ys = tf.ones((4, 2))

assert float(calls_override(est, xs)) == 12.0
assert float(calls_fit(model, xs, ys)) == 12.0
