import tensorflow as tf


def calls_replaced(m, t):
    return m.predict(t) + tf.reduce_sum(t)


def calls_fit(m, x, y):
    m.fit(x, y, epochs=1, verbose=0)
    return tf.reduce_sum(x)


inp = tf.keras.Input(shape=(3,))
out = tf.keras.layers.Dense(2)(inp)
model = tf.keras.Model(inp, out)
model.compile(optimizer="sgd", loss="mse")

xs = tf.ones((4, 3))
ys = tf.ones((4, 2))

assert float(calls_fit(model, xs, ys)) == 12.0

other = tf.keras.Model(inp, out)
other.predict = len
assert float(calls_replaced(other, xs)) == 16.0
assert float(tf.function(calls_replaced)(other, xs)) == 16.0
