import tensorflow as tf


# Receives each row of each window of a dataset of float32 vectors of four.
def consume(row):
    return row


rows = tf.data.Dataset.from_tensor_slices(tf.ones((8, 4)))

for window in rows.window(2):
    for row in window:
        consume(row)
