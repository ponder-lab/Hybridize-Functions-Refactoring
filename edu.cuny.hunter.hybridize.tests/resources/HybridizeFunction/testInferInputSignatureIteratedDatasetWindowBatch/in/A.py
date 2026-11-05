import tensorflow as tf


# Receives each batch of each window of a dataset of float32 vectors of four.
def consume(batch):
    return batch


rows = tf.data.Dataset.from_tensor_slices(tf.ones((8, 4)))

for window in rows.window(3):
    for batch in window.batch(2):
        consume(batch)
