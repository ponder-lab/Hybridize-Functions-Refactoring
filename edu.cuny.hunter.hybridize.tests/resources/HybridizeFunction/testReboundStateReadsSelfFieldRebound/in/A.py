# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1140

# A retrieval layer whose index method rebinds the candidates its call reads through self. A traced call keeps the first candidates.

import tensorflow as tf


class Streaming(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self._candidates = tf.zeros([1, 2])
        self._identifiers = tf.zeros([1], dtype=tf.int32)

    def index(self, candidates, identifiers):
        self._candidates = candidates
        self._identifiers = identifiers

    def call(self, queries):
        scores = tf.matmul(queries, self._candidates, transpose_b=True)
        best = tf.argmax(scores, axis=1)
        return tf.gather(self._identifiers, best)


streaming = Streaming()
for i in range(2):
    streaming.index(
        tf.fill([3, 2], i + 1.0), tf.range(3 * i, 3 * i + 3, dtype=tf.int32)
    )
    print(streaming(tf.ones([1, 2])))
