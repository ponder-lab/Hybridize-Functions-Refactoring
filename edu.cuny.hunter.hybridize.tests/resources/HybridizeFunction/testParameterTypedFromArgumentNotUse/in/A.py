import tensorflow as tf


def f(adj):
    if isinstance(adj, tf.SparseTensor):
        return tf.sparse.sparse_dense_matmul(adj, adj)
    else:
        return tf.linalg.matmul(adj, adj)


f(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
