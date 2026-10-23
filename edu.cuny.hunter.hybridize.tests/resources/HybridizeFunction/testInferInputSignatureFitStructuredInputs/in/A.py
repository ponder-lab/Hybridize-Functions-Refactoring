import tensorflow as tf


# Receives the tuple of float32 inputs Keras's default train step passes to `call`.
def consume(inputs):
    return tf.reduce_sum(inputs[0]) + tf.reduce_sum(inputs[1])


class Model(tf.keras.Model):

    def call(self, inputs):
        consume(inputs)
        x, mask = inputs
        return tf.reduce_sum(x, axis=1, keepdims=True) + tf.reduce_sum(
            mask, axis=1, keepdims=True
        )


# Receives the tuple of a float32 and an int32 input the same step passes to `call`.
def consume_mixed(inputs):
    return tf.reduce_sum(inputs[0]) + tf.cast(tf.reduce_sum(inputs[1]), tf.float32)


class MixedModel(tf.keras.Model):

    def call(self, inputs):
        consume_mixed(inputs)
        x, ids = inputs
        return tf.reduce_sum(x, axis=1, keepdims=True)


features = (tf.ones((8, 4)), tf.ones((8, 3)))
model = Model()
model.compile(optimizer="sgd", loss="mse")
model.fit(
    tf.data.Dataset.from_tensor_slices((features, tf.ones((8, 1)))).batch(2),
    epochs=1,
    verbose=0,
)

mixed = (tf.ones((8, 4)), tf.zeros((8, 3), dtype=tf.int32))
mixed_model = MixedModel()
mixed_model.compile(optimizer="sgd", loss="mse")
mixed_model.fit(
    tf.data.Dataset.from_tensor_slices((mixed, tf.ones((8, 1)))).batch(2),
    epochs=1,
    verbose=0,
)
