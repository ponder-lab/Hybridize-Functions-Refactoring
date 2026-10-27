import tensorflow as tf


# Receives the tuple of float32 inputs Keras's default train step passes to `call`, in batches of 2 and a final batch of 1.
def consume(inputs):
    return tf.reduce_sum(inputs[0]) + tf.reduce_sum(inputs[1])


class Model(tf.keras.Model):

    def call(self, inputs):
        consume(inputs)
        x, mask = inputs
        return tf.reduce_sum(x, axis=1, keepdims=True) + tf.reduce_sum(
            mask, axis=1, keepdims=True
        )


features = (tf.ones((7, 4)), tf.ones((7, 3)))
model = Model()
model.compile(optimizer="sgd", loss="mse")
model.fit(
    tf.data.Dataset.from_tensor_slices((features, tf.ones((7, 1)))).batch(2),
    epochs=1,
    verbose=0,
)
