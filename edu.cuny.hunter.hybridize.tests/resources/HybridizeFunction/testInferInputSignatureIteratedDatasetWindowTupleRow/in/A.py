import tensorflow as tf


# Receives each row of the first component's window of a dataset of pairs, a float32 vector of four and an int32 vector of three.
def consume(row):
    return row


pairs = tf.data.Dataset.from_tensor_slices(
    (tf.ones((8, 4)), tf.zeros((8, 3), tf.int32))
)

for a, b in pairs.window(2):
    for row in a:
        consume(row)
