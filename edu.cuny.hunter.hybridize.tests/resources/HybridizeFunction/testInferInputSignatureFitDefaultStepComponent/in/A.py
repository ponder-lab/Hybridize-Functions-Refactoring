import tensorflow as tf


# Receives the int32 component of the inputs Keras's default train step passes to `call`.
def count(index):
    return tf.cast(tf.reduce_sum(index), tf.float32)


class Model(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x, index = inputs
        return self.dense(x) + count(index)


features = (tf.ones((8, 4)), tf.zeros((8,), dtype=tf.int32))
dataset = tf.data.Dataset.from_tensor_slices((features, tf.ones((8, 1)))).batch(2)

model = Model()
model.compile(optimizer="sgd", loss="mse")
model.fit(dataset, epochs=1)
