# A function the program never calls, whose parameter is classified as a tensor from its type hint alone (#984). Having no call-graph
# node, it gets no tensor types. With the static-shape-read check forced on, it is checked anyway, and the check must pass determinately
# instead of failing the project.
import tensorflow as tf


def f(x: tf.Tensor):
    return tf.reshape(x, [x.shape[0], 1])
