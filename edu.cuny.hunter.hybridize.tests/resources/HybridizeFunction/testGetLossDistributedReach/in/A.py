import tensorflow as tf


class Model(tf.keras.Model):
    def call(self, x, training=True):
        return tf.cast(x, tf.int32), None

    def get_loss(self, real, pred):
        return tf.reduce_sum(real) + tf.reduce_sum(pred)

    def _train_step(self, inputs, targets):
        predictions, _ = self(inputs, training=True)
        return self.get_loss(targets, predictions)

    def distributed_train_step(self, strategy, inputs, targets):
        return strategy.run(self._train_step, args=(inputs, targets))


ds = tf.data.Dataset.from_tensor_slices(
    (
        tf.constant([[1, 2], [3, 4]], dtype=tf.int32),
        tf.constant([[5, 6], [7, 8]], dtype=tf.int32),
    )
)
ds = ds.padded_batch(2)

m = Model()
strategy = tf.distribute.MirroredStrategy()
dist_ds = strategy.experimental_distribute_dataset(ds)
for inp, tar in dist_ds:
    m.distributed_train_step(strategy, inp, tar)
