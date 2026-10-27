# Fixture for #976: `adjacency_lists` receives a bare edge tensor at one call site and a list of
# differently shaped edge tensors at another. The list makes the parameter a container, the mix
# leaves its element structure unmodeled, and the bare tensor gives it flat tensor evidence.
# Reducing that flat evidence would emit one `TensorSpec`, which the list caller cannot bind to, so
# the signature must drop with TENSOR_CONTAINER_UNSUPPORTED even though `node_embeddings` resolves.
# Reduces NLPGNN's `MessagePassing._calculate_type_to_incoming_edges_num`, which `GIN` calls with
# a bare edge tensor and `GraphSAGE` with a list of them.
import tensorflow as tf


def count_edges(node_embeddings, adjacency_lists):
    counts = []
    for adjacency_list in adjacency_lists:
        counts.append(tf.size(adjacency_list))
    return tf.reduce_sum(node_embeddings) + tf.cast(tf.add_n(counts), tf.float32)


embeddings = tf.ones([4, 16])

assert count_edges(embeddings, tf.zeros([3, 2], dtype=tf.int32)).shape == ()
assert (
    count_edges(
        embeddings, [tf.zeros([3, 2], dtype=tf.int32), tf.zeros([5, 2], dtype=tf.int32)]
    ).shape
    == ()
)
