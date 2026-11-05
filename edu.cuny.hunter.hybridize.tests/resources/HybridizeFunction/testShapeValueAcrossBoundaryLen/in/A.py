import tensorflow as tf


def add_self_loops(edges, num_nodes):
    loops = tf.range(num_nodes)
    return tf.gather(edges, loops)


def propagate(nodes, edges):
    return add_self_loops(edges, len(nodes))


def counted(nodes):
    return tf.range(len(nodes))


n = tf.ones([4, 3])
e = tf.ones([4, 2])
print(propagate(n, e).shape)
print(counted(n).shape)
