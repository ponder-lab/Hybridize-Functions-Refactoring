import tensorflow as tf


# Receives a pair of float32 inputs from one model and a pair of a float32 and an int32 input from the other.
def consume(inputs):
    return tf.reduce_sum(inputs[0]) + tf.cast(tf.reduce_sum(inputs[1]), tf.float32)


class Model(tf.keras.Model):

    def call(self, inputs):
        consume(inputs)
        x, mask = inputs
        return tf.reduce_sum(x, axis=1, keepdims=True)


class MixedModel(tf.keras.Model):

    def call(self, inputs):
        consume(inputs)
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
