# Fixture for #781's corpus shape, reduced from NLPGNN's Planetoid.load -> GATLayer chain: a
# loader builds a singleton list holding one edge-index tensor and returns it in a tuple; a
# Keras model method receives it (through the model-call trampoline) alongside a dense tensor
# and iterates it. The parameter classifies as a tensor container and reduces to a nested
# entry, yielding the two-entry signature
# `[tf.TensorSpec(shape=(2, 1), dtype=tf.float32), [tf.TensorSpec(shape=(2, 2), dtype=tf.int32)]]`.
import numpy as np
import tensorflow as tf


class Loader:

    def load(self):
        features = tf.constant(np.array([[1.0], [2.0]], dtype=np.float32))
        adj = tf.constant([[0, 1], [1, 0]], dtype=tf.int32)
        return features, [adj]


class Model(tf.keras.Model):

    def call(self, node_embeddings, adjacency_lists):
        total = tf.reduce_sum(node_embeddings)
        for adj in adjacency_lists:
            total = total + tf.cast(tf.reduce_sum(adj), tf.float32)
        return total


loader = Loader()
features, adjacency_lists = loader.load()
model = Model()
model(features, adjacency_lists)
