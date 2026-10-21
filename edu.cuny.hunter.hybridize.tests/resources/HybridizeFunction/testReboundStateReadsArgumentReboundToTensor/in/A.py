# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1140

# A function reading an argument that one caller may rebind to a tensor a helper returns. tf.function passes a tensor argument in afresh
# on every call, so nothing it reads through that argument is rebound state.

from typing import List, NamedTuple

import tensorflow as tf


class GraphInput(NamedTuple):
    node_embeddings: tf.Tensor
    adjacency_lists: List


def add_self_loops(adjacency_lists, num_nodes):
    loop_index = tf.tile(tf.expand_dims(tf.range(0, num_nodes), 1), [1, 2])
    row = adjacency_lists[:, 0]
    col = adjacency_lists[:, 1]
    mask = row != col
    return tf.concat([adjacency_lists[mask], loop_index], 0)


class MessagePassing(tf.keras.layers.Layer):
    def propagate(self, inputs):
        node_embeddings = inputs.node_embeddings
        adjacency_lists = inputs.adjacency_lists
        return self._calculate_messages(node_embeddings, adjacency_lists)

    def _count_incoming_edges(self, node_embeddings, adjacency_lists):
        counts = []
        for adjacency_list in adjacency_lists:
            targets = adjacency_list[:, 1]
            counts.append(
                tf.scatter_nd(
                    indices=tf.expand_dims(targets, -1),
                    updates=tf.ones_like(targets, dtype=tf.float32),
                    shape=(tf.shape(node_embeddings)[0],),
                )
            )
        return tf.stack(counts)


def calculate_messages(node_embeddings, adjacency_lists, counts):
    messages = []
    for i, adjacency_list in enumerate(adjacency_lists):
        sources = tf.gather(node_embeddings, adjacency_list[:, 0])
        weights = tf.gather(counts[i, :], adjacency_list[:, 1])
        messages.append(sources * tf.expand_dims(weights, -1))
    return tf.concat(messages, 0)


# Unless it concatenates, the caller rebinds the adjacency lists to the tensor a helper returns.
class SelfLoopConvolution(MessagePassing):
    def __init__(self, concat=True):
        super().__init__()
        self.concat = concat

    def call(self, inputs):
        node_embeddings = inputs.node_embeddings
        adjacency_lists = inputs.adjacency_lists
        if not self.concat:
            adjacency_lists = add_self_loops(adjacency_lists, len(node_embeddings))
        return self.propagate(GraphInput(node_embeddings, adjacency_lists))

    def _calculate_messages(self, node_embeddings, adjacency_lists):
        counts = self._count_incoming_edges(node_embeddings, adjacency_lists)
        return calculate_messages(node_embeddings, adjacency_lists, counts)


# The same, without the rebinding.
class PlainConvolution(MessagePassing):
    def call(self, inputs):
        return self.propagate(
            GraphInput(inputs.node_embeddings, inputs.adjacency_lists)
        )

    def _calculate_messages(self, node_embeddings, adjacency_lists):
        counts = self._count_incoming_edges(node_embeddings, adjacency_lists)
        return calculate_messages(node_embeddings, adjacency_lists, counts)


self_loop = SelfLoopConvolution()
plain = PlainConvolution()
for i in range(2):
    graph = GraphInput(
        tf.fill([3, 2], i + 1.0), [tf.constant([[0, 1], [1, 2]], dtype=tf.int32)]
    )
    print(self_loop(graph))
    print(plain(graph))
