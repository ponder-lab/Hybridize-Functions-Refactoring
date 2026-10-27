import tensorflow as tf


# Receives the output sequence of an `RNN` over a built-in cell.
def average(sequence):
    return tf.reduce_mean(sequence)


cell = tf.keras.layers.SimpleRNNCell(4)
outputs = tf.keras.layers.RNN(cell, return_sequences=True, return_state=True)(
    inputs=tf.ones((2, 3, 5)), initial_state=[tf.zeros((2, 4))]
)
average(outputs[0])
