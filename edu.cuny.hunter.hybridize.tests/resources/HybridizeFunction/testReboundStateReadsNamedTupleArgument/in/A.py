# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1140

# A NamedTuple argument's fields are not rebound state: tf.function flattens a NamedTuple argument into its tensors on every call.

from typing import List, NamedTuple

import tensorflow as tf


class GraphInput(NamedTuple):
    node_embeddings: tf.Tensor
    adjacency_lists: List


class AutoEncoder(tf.keras.Model):
    def call(self, inputs, training=None):
        node_embeddings = inputs.node_embeddings
        adjacency_list = inputs.adjacency_lists[0]
        sources = tf.gather(node_embeddings, adjacency_list[:, 0])
        targets = tf.gather(node_embeddings, adjacency_list[:, 1])
        return tf.reduce_sum(sources * targets, axis=-1)


model = AutoEncoder()
for i in range(2):
    graph = GraphInput(
        tf.fill([3, 2], i + 1.0), [tf.constant([[0, 1], [1, 2]], dtype=tf.int32)]
    )
    print(model(graph, training=False))
